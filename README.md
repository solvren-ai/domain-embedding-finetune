## Fine-Tuning BGE Embeddings Using Synthetic Data from Amazon Bedrock

### Getting Started

This guide assumes you have already set up a SageMaker Studio environment as described in the blog post. Follow these steps to set up your environment and run the code:

1. Clone the repository:

   ```bash
   git clone https://github.com/solvren-ai/domain-embedding-finetune.git
   cd domain-embedding-finetune
   ```

2. Create the Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

   This step may take several minutes to complete.

3. Activate the environment:

   ```bash
   conda init
   source ~/.bashrc 
   conda activate ft-embedding-blog
   ```

4. Add the new Conda environment to Jupyter:

   ```bash
   python -m ipykernel install --user --name=ft-embedding-blog
   ```

5. Open the Jupyter notebook:
   - From the SageMaker Studio Launcher, open the repository folder named `domain-embedding-finetune`.
   - Open the file `finetune-bge-embeddings.ipynb`.

6. Select the correct kernel:
   - From the "Kernel" dropdown menu in the notebook, select "Change Kernel...".
   - Choose "ft-embedding-blog".
   - If you don't see the kernel, try refreshing your browser.

7. You're now ready to run the code in the notebook. Follow the instructions in each cell to generate synthetic data, fine-tune the BGE model, evaluate its performance, and deploy it using Amazon SageMaker.

Note: Make sure you have the necessary permissions and quotas set up in your AWS account to use Amazon Bedrock and SageMaker services as described in the blog post.

## Run locally on Mac or on AWS

One entry point, [`run.py`](run.py), covers both modes (the notebook is unchanged). `--local` runs everything on Apple Silicon; without it, the steps run on AWS:

| Notebook (AWS) | `--local` | Cloud (default) |
|---|---|---|
| Uber / Lyft 10-K PDFs | HIPAA regulation text from eCFR (snapshot 2026-10-06). Train: 45 CFR Part 160 + Part 164 Subpart E (Privacy Rule). Val: Part 164 Subparts C (Security Rule) + D (Breach Notification). | same |
| Bedrock `meta.llama3-70b-instruct-v1:0` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` via `mlx-lm` (Apache-2.0) | Bedrock `amazon.nova-lite-v1:0` via the Converse API, same prompt |
| FlagEmbedding + torchrun on CUDA | sentence-transformers `MultipleNegativesRankingLoss` on MPS | the same training function in a SageMaker training job (Hugging Face PyTorch DLC, `ml.g5.xlarge`) |
| Hit rate + `InformationRetrievalEvaluator`; Titan reference | `InformationRetrievalEvaluator` (accuracy@k = hit rate@k, NDCG@10, MRR@10) over the full 304-chunk corpus and the 33 val chunks, at seq len 256 and 512; references: `bge-large-en-v1.5`, `snowflake-arctic-embed-m-v1.5`; general-domain check on BEIR SciFact | the same, run inside the training job after training |

Requirements: [uv](https://docs.astral.sh/uv/). For `--local`: an Apple Silicon Mac with 16 GB+ RAM and about 5 GB of disk for the models. Dependencies are declared inline in the script (PEP 723) and pinned in `run.py.lock`.

### Commands

```bash
git clone https://github.com/solvren-ai/domain-embedding-finetune.git
cd domain-embedding-finetune
uv run run.py selftest                # offline checks: parser, chunker, mode dispatch

# Local (Apple Silicon)
uv run run.py corpus --local          # download eCFR XML to data/raw/, write data/corpus_{train,val}.jsonl
uv run run.py generate --local        # MLX Qwen3-4B -> data/pairs_{train,val}.jsonl (~28 min on an M5, resumable)
uv run run.py train --local           # fine-tune on MPS -> models/bge-base-en-v1.5-hipaa/ (refuses to overwrite trained weights)
uv run run.py eval --local            # all models x both corpora x seq 256/512 -> results/metrics.json
uv run run.py beir --local            # BEIR SciFact check -> results/beir_scifact.json
uv run run.py all --local             # the five steps above

# Cloud (AWS); add --dry-run to any of these to print the requests and send nothing
uv run run.py corpus                  # same as local
uv run run.py generate                # Bedrock Nova Lite (Converse) -> data/pairs_*.jsonl
uv run run.py train --role arn:aws:iam::<acct>:role/<SageMakerRole>
                                      # upload code + data to S3, start a SageMaker job that runs train, eval and beir
uv run run.py all --role ...          # corpus, generate, train (eval and beir run inside the job)
uv run run.py all --dry-run           # print the Bedrock and SageMaker requests

# Publish (either mode) to the SolvrenAI Hugging Face org: creates missing repos as private, refuses to upload into a public repo
uv run run.py publish --dry-run
uv run run.py publish                 # HF_TOKEN (write access to SolvrenAI) from the env or --env-file (default ../.env)
```

Shared flags:
- `--local`: run on this machine.
- `--dry-run`: cloud steps and publish only; prints the requests and sends nothing.
- `--model`: the LLM for `generate`, or the base embedding model for `train` and `all`.
- `--region`: defaults to `us-east-1`.
- `--instance-type`: defaults to `ml.g5.xlarge`.
- Also `--role` (or the `SAGEMAKER_ROLE_ARN` environment variable) and `--bucket` (default `sagemaker-<region>-<account>`).
- Smoke-run flags: `--max-chunks N` (generate) and `--max-steps N` (train).

`generate` is resumable: it skips chunks that already have pairs. The committed `data/pairs_*.jsonl` cover every chunk, so delete them first to generate new pairs (locally or with Bedrock); otherwise `generate` and `all` reuse the published pairs.

In cloud mode, `eval` and `beir` don't run on their own: they run inside the SageMaker training job after training. The fine-tuned model and `results/` come back in the job's `model.tar.gz` on S3.

Cloud prerequisites:
- **AWS credentials** that boto3 can find (`aws configure`, `AWS_PROFILE` or environment variables). `--dry-run` needs none.
- **Bedrock:** model access to Amazon Nova Lite (`amazon.nova-lite-v1:0`) enabled in the target region, plus `bedrock:InvokeModel` permission for your credentials. Outside `us-east-1` you may need the cross-region inference profile: `generate --model us.amazon.nova-lite-v1:0`.
- **SageMaker execution role:** trusted by `sagemaker.amazonaws.com`, with read/write access to the S3 bucket. It also needs CloudWatch Logs and ECR pull permissions, which `AmazonSageMakerFullAccess` covers. Your own credentials need `sagemaker:CreateTrainingJob`, `iam:PassRole` on that role, and S3 write access.
- **S3 bucket:** `--bucket`, or the default `sagemaker-<region>-<account>` must already exist (the SageMaker Python SDK or Studio creates it on first use; `run.py` does not).
- **GPU quota:** the "ml.g5.xlarge for training job usage" Service Quota must be at least 1 in the region (new accounts often start at 0).
- **Network:** the job downloads the base and reference models from the Hugging Face Hub, so it needs internet access (the default, no VPC).

Outputs:
- `data/pairs_*.jsonl`: the synthetic dataset, one `{"query", "pos", "chunk_id", "split"}` per line (FlagEmbedding-compatible).
- `results/metrics.json`, `results/beir_scifact.json`, `results/train_info.json`: the evaluation metrics and training run details. Headline: NDCG@10 over the full corpus goes from 0.834 (base) to 0.907 (fine-tuned); SciFact is unchanged (0.740 to 0.743). Full tables are in [`MODEL_CARD.md`](MODEL_CARD.md).
- `models/bge-base-en-v1.5-hipaa/`: the fine-tuned weights (git-ignored). See [`MODEL_CARD.md`](MODEL_CARD.md) and [`DATASET_CARD.md`](DATASET_CARD.md). Published as [SolvrenAI/bge-base-en-v1.5-hipaa](https://huggingface.co/SolvrenAI/bge-base-en-v1.5-hipaa) and [SolvrenAI/hipaa-retrieval-pairs](https://huggingface.co/datasets/SolvrenAI/hipaa-retrieval-pairs).

Downloaded regulation XML goes in `data/raw/` (git-ignored). It is only parsed, never executed. The notebook's hard-negative mining, LM-Cocktail merging and SageMaker endpoint deployment are not ported.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
