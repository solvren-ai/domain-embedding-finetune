---
# DRAFT - not pushed to the Hugging Face Hub.
language: en
license: mit
base_model: BAAI/bge-base-en-v1.5
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
      - task: {type: information-retrieval}
        dataset: {name: HIPAA Security + Breach Notification synthetic queries (val), type: custom}
        metrics:
          - {type: ndcg_at_10, value: 0.9181}
          - {type: mrr_at_10, value: 0.8908}
          - {type: accuracy_at_1, value: 0.8182}
---

# bge-base-en-v1.5-hipaa (draft)

`BAAI/bge-base-en-v1.5` fine-tuned for retrieval over US healthcare privacy and security regulation (HIPAA, 45 CFR Parts 160 and 164). It is meant as the retriever in RAG systems that answer compliance questions over regulatory text.

This is a showcase from Solvren AI (AI forward deployment for regulated industries). It was trained end to end on a MacBook (Apple M5, 16 GB): local data generation, local training, no cloud services. The pipeline is [`local/run_local.py`](local/run_local.py) in the `mac-local` branch of [solvren-ai/finetune-bge-embeddings-blog](https://github.com/solvren-ai/finetune-bge-embeddings-blog), adapted from [aws-samples/finetune-bge-embeddings-blog](https://github.com/aws-samples/finetune-bge-embeddings-blog) (MIT-0).

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/bge-base-en-v1.5-hipaa")  # local path; Hub id TBD
q = model.encode(["Represent this sentence for searching relevant passages: "
                  "When must a covered entity notify the media of a breach?"])
p = model.encode(["§ 164.406 Notification to the media. ..."])
print(model.similarity(q, p))
```

Prefix queries with the standard BGE instruction `Represent this sentence for searching relevant passages: ` (the model was trained with it). Don't prefix passages.

## Training data

| Split | Source (eCFR, snapshot 2026-10-06) | Chunks | Query-passage pairs |
|---|---|---|---|
| train | [45 CFR Part 160](https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=160) (HIPAA general administrative requirements) + [45 CFR Part 164 Subpart E](https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164&subpart=E) (Privacy Rule) | 271 | 813 |
| val | [45 CFR Part 164 Subpart C](https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164&subpart=C) (Security Rule) + [Subpart D](https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164&subpart=D) (Breach Notification Rule) | 33 | 99 |

- The regulation text is a US federal government work (public domain).
- Chunks are whole paragraphs packed up to about 220 words within one CFR section, each prefixed with the section heading.
- Up to 3 synthetic questions per chunk were generated locally with [`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) (Apache-2.0) via `mlx-lm`, using the original notebook's prompt (temperature 0.7, top-p 0.8).
- Train and val come from different regulations, so the evaluation is out-of-document.
- The dataset ships in the repo as `data/pairs_train.jsonl` and `data/pairs_val.jsonl` (`{"query", "pos", "chunk_id", "split"}`).

## Training procedure

- sentence-transformers `MultipleNegativesRankingLoss` (in-batch negatives), with a `NO_DUPLICATES` batch sampler.
- 4 epochs, batch size 8, learning rate 2e-5, 10% warmup, max sequence length 256, fp32.
- Device: Apple MPS. 408 steps in 7.6 minutes.
- Not done (unlike the original notebook): hard-negative mining, LM-Cocktail model merging.

## Evaluation

`InformationRetrievalEvaluator` on the val split: 99 queries against all 33 val chunks, cosine similarity, same query instruction for both models. Hit rate@k = accuracy@k.

| Metric | bge-base-en-v1.5 | **fine-tuned** |
|---|---|---|
| hit rate@1 | 0.7576 | **0.8182** |
| hit rate@3 | 0.9192 | **0.9697** |
| hit rate@5 | 0.9495 | **0.9798** |
| hit rate@10 | 0.9899 | **1.0000** |
| NDCG@10 | 0.8796 | **0.9181** |
| MRR@10 | 0.8438 | **0.8908** |

Raw numbers: `results/metrics.json`.

## Limitations

- The val corpus is small (33 chunks), so absolute scores are optimistic compared with a full-size regulatory corpus. A 1-query change moves hit rate@1 by about 1 point. Treat the deltas as directional.
- The queries are LLM-generated from the passage they target, so they share vocabulary with it. Real user queries are messier.
- Coverage is HIPAA only. The model hasn't been tested on other healthcare regulation (e.g. 42 CFR Part 2, state privacy law) or on other domains, and general-domain retrieval may have regressed. Evaluate on your own corpus before use.
- The base model was evaluated at its default max length (512) and the fine-tuned model at 256.
- Retrieval output isn't legal advice.
