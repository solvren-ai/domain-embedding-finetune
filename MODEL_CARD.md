---
language: en
license: mit
base_model: BAAI/bge-base-en-v1.5
datasets:
  - SolvrenAI/hipaa-retrieval-pairs
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - sentence-transformers
  - feature-extraction
  - retrieval
  - rag
  - healthcare
  - hipaa
  - regulatory
model-index:
  - name: bge-base-en-v1.5-hipaa
    results:
      - task: {type: retrieval}
        dataset: {name: HIPAA val queries over full 304-chunk corpus, type: SolvrenAI/hipaa-retrieval-pairs, split: validation}
        metrics:
          - {type: ndcg_at_10, value: 0.9071}
          - {type: mrr_at_10, value: 0.8759}
          - {type: accuracy_at_1, value: 0.7879}
          - {type: accuracy_at_10, value: 1.0}
      - task: {type: retrieval}
        dataset: {name: HIPAA val queries over 33 val chunks, type: SolvrenAI/hipaa-retrieval-pairs, split: validation}
        metrics:
          - {type: ndcg_at_10, value: 0.9256}
          - {type: mrr_at_10, value: 0.9004}
          - {type: accuracy_at_1, value: 0.8283}
      - task: {type: retrieval}
        dataset: {name: BEIR SciFact, type: mteb/scifact, split: test}
        metrics:
          - {type: ndcg_at_10, value: 0.7427}
---

# bge-base-en-v1.5-hipaa

`BAAI/bge-base-en-v1.5` fine-tuned for retrieval over US healthcare privacy and security regulation (HIPAA, 45 CFR Parts 160 and 164). It is meant as the retriever in RAG systems that answer compliance questions over regulatory text.

This is a showcase from Solvren AI (AI forward deployment for regulated industries). It was trained end to end on a MacBook (Apple M5, 16 GB): local data generation, local training, no cloud services. The pipeline is [`local/run_local.py`](https://github.com/solvren-ai/domain-embedding-finetune/blob/mac-local/local/run_local.py) on the `mac-local` branch of [solvren-ai/domain-embedding-finetune](https://github.com/solvren-ai/domain-embedding-finetune), adapted from [aws-samples/finetune-bge-embeddings-blog](https://github.com/aws-samples/finetune-bge-embeddings-blog) (MIT-0).

## Results

The headline eval uses all 99 held-out queries, each with one relevant chunk. They come from the HIPAA Security Rule and Breach Notification Rule, regulations the model never saw in training. Each query retrieves over the **full 304-chunk corpus** (train and val chunks), so the 271 Privacy Rule and Part 160 chunks act as same-domain distractors. All models use the same query instruction, cosine similarity and `max_seq_length`. Hit rate@k is accuracy@k.

**Full corpus (304 chunks), max_seq_length 512**

| Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|
| bge-base-en-v1.5 (base) | 0.6970 | 0.8788 | 0.9192 | 0.9495 | 0.8337 | 0.7956 |
| bge-large-en-v1.5 | 0.7475 | 0.8990 | 0.9495 | 0.9697 | 0.8638 | 0.8291 |
| snowflake-arctic-embed-m-v1.5 | 0.7778 | 0.8990 | 0.9192 | 0.9394 | 0.8649 | 0.8401 |
| **bge-base-en-v1.5-hipaa (this model)** | **0.7879** | **0.9596** | **0.9798** | **1.0000** | **0.9071** | **0.8759** |

**Val chunks only (33 chunks), max_seq_length 512**

| Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|
| bge-base-en-v1.5 (base) | 0.7576 | 0.9192 | 0.9495 | 0.9899 | 0.8796 | 0.8438 |
| bge-large-en-v1.5 | 0.7980 | 0.9293 | 0.9697 | 0.9899 | 0.9023 | 0.8732 |
| snowflake-arctic-embed-m-v1.5 | **0.8283** | 0.9293 | 0.9495 | 0.9899 | 0.9092 | 0.8836 |
| **bge-base-en-v1.5-hipaa (this model)** | **0.8283** | **0.9697** | **0.9899** | **1.0000** | **0.9256** | **0.9004** |

**NDCG@10 at max_seq_length 256** (the training length)

| Model | Full corpus | Val only |
|---|---|---|
| bge-base-en-v1.5 (base) | 0.8282 | 0.8779 |
| bge-large-en-v1.5 | 0.8581 | 0.8894 |
| snowflake-arctic-embed-m-v1.5 | 0.8551 | 0.9039 |
| **bge-base-en-v1.5-hipaa** | **0.8965** | **0.9181** |

**General-domain check: BEIR SciFact** (test split, 300 queries over 5,183 abstracts, max_seq_length 512)

| Model | NDCG@10 | Hit@10 |
|---|---|---|
| bge-base-en-v1.5 (base) | 0.7404 | 0.8833 |
| bge-base-en-v1.5-hipaa | 0.7427 (+0.0023) | 0.8767 (-0.0067) |

SciFact shows no meaningful regression; the deltas are within noise. This is one small task, not a full MTEB run.

The reference models were not fine-tuned. Both use the BGE query instruction their model cards recommend. Licenses: bge-large-en-v1.5 is MIT, snowflake-arctic-embed-m-v1.5 is Apache-2.0. Raw numbers are in `results/metrics.json` and `results/beir_scifact.json` in the GitHub repo.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("SolvrenAI/bge-base-en-v1.5-hipaa")
queries = ["When must a covered entity notify the media about a breach of unsecured PHI?"]
passages = [
    "§ 164.406 Notification to the media. (a) Standard. For a breach of unsecured protected health information "
    "involving more than 500 residents of a State or jurisdiction, a covered entity shall ... notify prominent "
    "media outlets serving the State or jurisdiction.",
    "§ 164.312 Technical safeguards. A covered entity or business associate must ... implement technical policies "
    "and procedures for electronic information systems that maintain electronic protected health information.",
]
q = model.encode(queries, prompt="Represent this sentence for searching relevant passages: ")
p = model.encode(passages)
print(model.similarity(q, p))  # the first passage scores higher
```

Prefix queries with the BGE instruction `Represent this sentence for searching relevant passages: ` (the model was trained with it). Don't prefix passages.

- The weights were saved with sentence-transformers 6.1, so loading them with sentence-transformers needs version 6.1 or later.
- The default `max_seq_length` is 512, which matches the headline results. Training used 256.
- With plain `transformers` (`AutoModel`), use CLS pooling followed by L2 normalization, as for the base BGE model.

## Training data

Dataset: [SolvrenAI/hipaa-retrieval-pairs](https://huggingface.co/datasets/SolvrenAI/hipaa-retrieval-pairs).

| Split | Source (eCFR, snapshot 2026-10-06) | Chunks | Query-passage pairs |
|---|---|---|---|
| train | [45 CFR Part 160](https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=160) (HIPAA general administrative requirements) + [45 CFR Part 164 Subpart E](https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164&subpart=E) (Privacy Rule) | 271 | 813 |
| validation | [45 CFR Part 164 Subpart C](https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164&subpart=C) (Security Rule) + [Subpart D](https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164&subpart=D) (Breach Notification Rule) | 33 | 99 |

- The regulation text is a US federal government work (public domain).
- Chunks are whole paragraphs packed up to about 220 words within one CFR section, each prefixed with the section heading.
- Up to 3 synthetic questions per chunk were generated locally with [`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) (Apache-2.0) via `mlx-lm`, using the original notebook's prompt (temperature 0.7, top-p 0.8).
- Train and validation come from different regulations.

## Training procedure

- sentence-transformers `MultipleNegativesRankingLoss` (in-batch negatives), with a `NO_DUPLICATES` batch sampler.
- 4 epochs, batch size 8, learning rate 2e-5, 10% warmup, max sequence length 256, fp32.
- Device: Apple MPS. 408 steps in 7.6 minutes.
- Not done (unlike the original notebook): hard-negative mining, LM-Cocktail model merging.

## Limitations

- The validation set is small: 99 queries. One query moves hit rate@1 by about 1 point, so treat the deltas as directional, not definitive.
- The queries are LLM-generated from the chunk they target, so they share its vocabulary. Real user queries are messier, and absolute scores will be lower on them.
- In the full-corpus eval, only the source chunk counts as relevant. A Privacy Rule chunk that also answers a query counts as a miss, so all models are scored equally strictly.
- This model was trained on the 271 train chunks that serve as distractors in the full-corpus eval (it never saw the val chunks or val queries). The reference models were not.
- Coverage is HIPAA only. The model hasn't been tested on other healthcare regulation (e.g. 42 CFR Part 2, state privacy law). The general-domain check covers one BEIR task. Evaluate on your own corpus before use.
- Retrieval output isn't legal advice. The eCFR is an authoritative but unofficial editorial compilation of the CFR.
