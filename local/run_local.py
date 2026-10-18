# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "mlx-lm>=0.28",
#   "sentence-transformers>=5.1",
#   "torch>=2.8",
#   "transformers>=4.56",
#   "accelerate>=1.10",
#   "datasets>=4.0",
# ]
# ///
"""Mac-local port of finetune-bge-embeddings.ipynb (no AWS).

Steps (run in order, or `all`):
  corpus    download HIPAA regulation text from eCFR, chunk it (train/val = different documents)
  generate  synthetic question->chunk pairs with a local MLX LLM (replaces Bedrock Llama 3)
  train     fine-tune BAAI/bge-base-en-v1.5 on MPS with MultipleNegativesRankingLoss
  eval      base vs fine-tuned vs references on HIPAA val queries (full + val-only corpus, seq 256/512)
  beir      general-domain regression check on BEIR SciFact (NDCG@10)
  selftest  offline checks for the parsing/chunking logic

Run from the repo root:  uv run local/run_local.py all
"""

import gzip
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DATA = Path("data")
RAW = DATA / "raw"  # downloaded regulation XML: untrusted input, parsed only, never executed
MODEL_OUT = Path("models/bge-base-en-v1.5-hipaa")
RESULTS = Path("results")

BASE_MODEL = "BAAI/bge-base-en-v1.5"
LLM = "mlx-community/Qwen3-4B-Instruct-2507-4bit"  # Apache-2.0
# Official BGE v1.5 query instruction; used for base and fine-tuned alike so the comparison is fair.
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

# eCFR snapshot pinned for reproducibility. Train and val are different regulations.
ECFR = "https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part={part}{sub}"
SOURCES = {
    "train": [  # HIPAA Administrative (Part 160) + Privacy Rule (Part 164 Subpart E)
        ("45cfr160", ECFR.format(part=160, sub="")),
        ("45cfr164E", ECFR.format(part=164, sub="&subpart=E")),
    ],
    "val": [  # HIPAA Security Rule (Subpart C) + Breach Notification Rule (Subpart D)
        ("45cfr164C", ECFR.format(part=164, sub="&subpart=C")),
        ("45cfr164D", ECFR.format(part=164, sub="&subpart=D")),
    ],
}

CHUNK_WORDS = 220  # ~300 BERT tokens, fits bge's 512 limit with the heading prefix
MIN_LEN = 100  # drop "[Reserved]"-style stubs (notebook used 20 chars)
NUM_QUESTIONS = 3

# Prompt kept from the notebook (cell 10); the "Assistant:" priming line is dropped since we use a chat template.
QA_PROMPT = """You are an AI assistant helping a teacher generate questions for an upcoming quiz. I will provide you with some context material in <context> tags. Your task is to generate {num} question(s) based only on the given context. The questions should be diverse and cover different parts of the context. Do not rely on any outside knowledge.

Here are some important guidelines to follow:
- Number each question, starting from 1.
- Put each question on its own line, with no blank lines in between.
- Do not generate any multiple choice questions.
- Restrict the questions to information provided in the context only.
- Do not reference the text, just directly ask a question that can be answered with it.
    - For example, do not add phrases like, "as referenced in the context" or "according to this section"
- Phrase questions the way a compliance officer, privacy officer, or health IT engineer would search for the answer.
- Do not output anything other than the questions.

Here is the context to use for generating questions:
<context>
{context}
</context>

Please generate your numbered questions in the following format:

1. Question 1
2. Question 2
3. Question 3

The number of questions should match the {num} value exactly."""


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def write_jsonl(p, rows):
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def fetch(name, url):
    out = RAW / f"{name}.xml"
    if not out.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "finetune-bge-local"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
        out.write_bytes(body)
    return out


def sections(xml_bytes):
    """Yield (heading, [paragraph, ...]) for each § section in an eCFR XML document."""
    root = ET.fromstring(xml_bytes)
    for div in root.iter("DIV8"):
        head = " ".join("".join(div.find("HEAD").itertext()).split()) if div.find("HEAD") is not None else ""
        paras = [" ".join("".join(p.itertext()).split()) for p in div.iter("P")]
        yield head, [p for p in paras if p]


def chunk_section(head, paras, max_words=CHUNK_WORDS):
    """Pack whole paragraphs into chunks of <= max_words (a single longer paragraph becomes its own chunk)."""
    chunks, cur = [], []
    for p in paras:
        if cur and sum(len(x.split()) for x in cur) + len(p.split()) > max_words:
            chunks.append(cur)
            cur = []
        cur.append(p)
    if cur:
        chunks.append(cur)
    return [head + "\n" + "\n".join(c) for c in chunks]


def parse_questions(text, n=NUM_QUESTIONS):
    qs = [re.sub(r"^\s*\d+[\).\s]\s*", "", line).strip() for line in text.strip().split("\n")]
    return [q for q in qs if len(q) > 10 and q.endswith("?")][:n]


def step_corpus():
    for split, srcs in SOURCES.items():
        rows = []
        for name, url in srcs:
            for head, paras in sections(fetch(name, url).read_bytes()):
                for i, text in enumerate(chunk_section(head, paras)):
                    if len(text) >= MIN_LEN:
                        sec = head.split()[1] if head.startswith("§") else name
                        rows.append({"id": f"{sec}#{i}", "source": url, "section": head, "text": text})
        assert len({r["id"] for r in rows}) == len(rows), "duplicate chunk ids"
        write_jsonl(DATA / f"corpus_{split}.jsonl", rows)
        print(f"{split}: {len(rows)} chunks from {[n for n, _ in srcs]}")


def step_generate():
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(LLM)
    sampler = make_sampler(temp=0.7, top_p=0.8)  # Qwen3-Instruct-2507 recommended settings
    for split in ("train", "val"):
        out = DATA / f"pairs_{split}.jsonl"
        done = {r["chunk_id"] for r in read_jsonl(out)} if out.exists() else set()  # resumable
        corpus = [c for c in read_jsonl(DATA / f"corpus_{split}.jsonl") if c["id"] not in done]
        t0 = time.time()
        with out.open("a") as f:
            for i, c in enumerate(corpus):
                msgs = [{"role": "user", "content": QA_PROMPT.format(num=NUM_QUESTIONS, context=c["text"])}]
                prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
                resp = generate(model, tok, prompt, max_tokens=256, sampler=sampler)
                for q in parse_questions(resp):
                    # FlagEmbedding-compatible {"query", "pos"} plus provenance fields
                    f.write(json.dumps({"query": q, "pos": [c["text"]], "chunk_id": c["id"], "split": split}) + "\n")
                f.flush()
                if i % 20 == 0:
                    print(f"{split} {i + 1}/{len(corpus)} chunks, {time.time() - t0:.0f}s", flush=True)
        print(f"{split}: {len(read_jsonl(out))} pairs")
    del model
    import mlx.core as mx
    mx.clear_cache()  # release the LLM before training when run via `all`


def step_train(epochs=4, batch_size=8, lr=2e-5):  # small batch: 16 GB unified memory shared with other MLX jobs
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    assert torch.backends.mps.is_available(), "MPS not available"
    pairs = read_jsonl(DATA / "pairs_train.jsonl")
    ds = Dataset.from_dict({"anchor": [p["query"] for p in pairs], "positive": [p["pos"][0] for p in pairs]})
    model = SentenceTransformer(BASE_MODEL, device="mps")
    model.max_seq_length = 256  # as the notebook (passage_max_len 256); trims the tail of the longest chunks
    args = SentenceTransformerTrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        # several questions share a chunk; no-duplicates keeps them out of each other's in-batch negatives
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={"anchor": QUERY_PROMPT},
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=42,
    )
    t0 = time.time()
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=MultipleNegativesRankingLoss(model)).train()
    minutes = (time.time() - t0) / 60
    model.save(str(MODEL_OUT))
    RESULTS.mkdir(exist_ok=True)
    meta = {"train_pairs": len(pairs), "epochs": epochs, "batch_size": batch_size, "lr": lr, "train_minutes": round(minutes, 1)}
    (RESULTS / "train_info.json").write_text(json.dumps(meta, indent=2))
    print(meta)


# Off-the-shelf references (no fine-tuning). Both use the same query instruction as BGE v1.5.
REFERENCE_MODELS = {
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",  # MIT
    "snowflake-arctic-embed-m-v1.5": "Snowflake/snowflake-arctic-embed-m-v1.5",  # Apache-2.0
}
METRIC_KEYS = {"accuracy@1": "hit_rate@1", "accuracy@3": "hit_rate@3", "accuracy@5": "hit_rate@5",
               "accuracy@10": "hit_rate@10", "ndcg@10": "ndcg@10", "mrr@10": "mrr@10", "map@10": "map@10"}
ENCODE_BATCH = 16  # small: unified memory is shared with other MLX jobs


def ir_eval(model, name, queries, corpus, relevant):
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    ev = InformationRetrievalEvaluator(
        queries, corpus, relevant, name=name, query_prompt=QUERY_PROMPT, batch_size=ENCODE_BATCH,
        accuracy_at_k=[1, 3, 5, 10], precision_recall_at_k=[1, 5, 10], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
    )
    m = ev(model)
    return {v: round(m[f"{name}_cosine_{k}"], 4) for k, v in METRIC_KEYS.items()}


def step_eval():
    """HIPAA val queries vs (a) the full 304-chunk corpus (train chunks as distractors) and (b) the 33 val chunks,
    for every model at max_seq_length 256 and 512."""
    from sentence_transformers import SentenceTransformer

    train_c, val_c = read_jsonl(DATA / "corpus_train.jsonl"), read_jsonl(DATA / "corpus_val.jsonl")
    assert not {c["id"] for c in train_c} & {c["id"] for c in val_c}, "train/val chunk ids overlap"
    pairs = read_jsonl(DATA / "pairs_val.jsonl")
    queries = {f"q{i}": p["query"] for i, p in enumerate(pairs)}
    relevant = {f"q{i}": {p["chunk_id"]} for i, p in enumerate(pairs)}
    corpora = {"full_corpus": {c["id"]: c["text"] for c in train_c + val_c}, "val_only": {c["id"]: c["text"] for c in val_c}}
    models = {"bge-base-en-v1.5": BASE_MODEL, "bge-base-en-v1.5-hipaa": str(MODEL_OUT), **REFERENCE_MODELS}
    out = {"val_queries": len(queries), "corpus_size": {k: len(v) for k, v in corpora.items()}, "query_prompt": QUERY_PROMPT,
           "results": {k: {"256": {}, "512": {}} for k in corpora}}
    for mname, path in models.items():
        model = SentenceTransformer(path, device="mps")
        for seq in (256, 512):
            model.max_seq_length = seq
            for cname, corpus in corpora.items():
                out["results"][cname][str(seq)][mname] = r = ir_eval(model, f"{cname}-{seq}", queries, corpus, relevant)
                print(cname, seq, mname, r, flush=True)
        del model
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))


def step_beir():
    """General-domain regression check: BEIR SciFact test split (300 queries, 5,183 abstracts), NDCG@10."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    corpus = {r["_id"]: (r["title"] + " " + r["text"]).strip() for r in load_dataset("mteb/scifact", "corpus", split="corpus")}
    qrels = {}
    for r in load_dataset("mteb/scifact", "default", split="test"):
        if r["score"] > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    queries = {r["_id"]: r["text"] for r in load_dataset("mteb/scifact", "queries", split="queries") if r["_id"] in qrels}
    out = {"task": "BEIR SciFact (mteb/scifact, test)", "queries": len(queries), "corpus": len(corpus), "max_seq_length": 512}
    for mname, path in (("bge-base-en-v1.5", BASE_MODEL), ("bge-base-en-v1.5-hipaa", str(MODEL_OUT))):
        model = SentenceTransformer(path, device="mps")
        model.max_seq_length = 512
        out[mname] = ir_eval(model, "scifact", queries, corpus, qrels)
        print(mname, out[mname], flush=True)
        del model
    out["ndcg@10_delta"] = round(out["bge-base-en-v1.5-hipaa"]["ndcg@10"] - out["bge-base-en-v1.5"]["ndcg@10"], 4)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "beir_scifact.json").write_text(json.dumps(out, indent=2))


def step_selftest():
    xml = b'<DIV6><DIV8 TYPE="SECTION"><HEAD>\xc2\xa7 164.302 Applicability.</HEAD><P>(a) A <I>covered</I> entity.</P><P/></DIV8></DIV6>'
    assert list(sections(xml)) == [("§ 164.302 Applicability.", ["(a) A covered entity."])]
    cs = chunk_section("H", ["a " * 150, "b " * 150, "c " * 10])
    assert len(cs) == 2 and cs[0].startswith("H\na") and cs[1].count("\n") == 2
    assert parse_questions("1. What is PHI?\n2) Who must comply?\nHere you go\n3. Is it ok\n4. Extra one here?") == [
        "What is PHI?", "Who must comply?", "Extra one here?"]
    print("selftest ok")


STEPS = {"corpus": step_corpus, "generate": step_generate, "train": step_train, "eval": step_eval, "beir": step_beir, "selftest": step_selftest}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    for n in (["corpus", "generate", "train", "eval", "beir"] if names == ["all"] else names):
        STEPS[n]()
