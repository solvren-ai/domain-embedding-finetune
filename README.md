## Fine-Tuning BGE Embeddings Using Synthetic Data from Amazon Bedrock

### Getting Started

This guide assumes you have already set up a SageMaker Studio environment as described in the blog post. Follow these steps to set up your environment and run the code:

1. Clone the repository:

   ```bash
   git clone https://github.com/aws-samples/finetune-bge-embeddings-blog.git
   cd finetune-bge-embeddings-blog
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
   - From the SageMaker Studio Launcher, open the repository folder named `finetune-bge-embeddings-blog`.
   - Open the file `finetune-bge-embeddings.ipynb`.

6. Select the correct kernel:
   - From the "Kernel" dropdown menu in the notebook, select "Change Kernel...".
   - Choose "ft-embedding-blog".
   - If you don't see the kernel, try refreshing your browser.

7. You're now ready to run the code in the notebook. Follow the instructions in each cell to generate synthetic data, fine-tune the BGE model, evaluate its performance, and deploy it using Amazon SageMaker.

Note: Make sure you have the necessary permissions and quotas set up in your AWS account to use Amazon Bedrock and SageMaker services as described in the blog post.

## Run locally on Mac

A no-AWS port of the notebook lives in [`local/run_local.py`](local/run_local.py) (the notebook is unchanged). It runs fully on Apple Silicon:

| Notebook (AWS) | Local port |
|---|---|
| Uber / Lyft 10-K PDFs | HIPAA regulation text from eCFR (snapshot 2026-10-06). Train: 45 CFR Part 160 + Part 164 Subpart E (Privacy Rule). Val: Part 164 Subparts C (Security Rule) + D (Breach Notification). |
| Bedrock `meta.llama3-70b-instruct-v1:0` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` via `mlx-lm` (Apache-2.0) |
| FlagEmbedding + torchrun on CUDA | sentence-transformers `MultipleNegativesRankingLoss` on MPS |
| Hit rate + `InformationRetrievalEvaluator` | `InformationRetrievalEvaluator` (accuracy@k = hit rate@k, NDCG@10, MRR@10) |

Requirements: an Apple Silicon Mac with 16 GB+ RAM, [uv](https://docs.astral.sh/uv/), and about 5 GB of disk for the models. Dependencies are declared inline in the script (PEP 723) and pinned in `local/run_local.py.lock`.

```bash
git clone -b mac-local https://github.com/solvren-ai/finetune-bge-embeddings-blog.git
cd finetune-bge-embeddings-blog
uv run local/run_local.py selftest   # offline parser/chunker checks
uv run local/run_local.py corpus     # download eCFR XML to data/raw/, write data/corpus_{train,val}.jsonl
uv run local/run_local.py generate   # write data/pairs_{train,val}.jsonl (~15 min, resumable)
uv run local/run_local.py train      # fine-tune on MPS, save to models/bge-base-en-v1.5-hipaa/
uv run local/run_local.py eval       # base vs fine-tuned, write results/metrics.json
# or all four steps in one go: uv run local/run_local.py all
```

Outputs:
- `data/pairs_*.jsonl`: the synthetic dataset, one `{"query", "pos", "chunk_id", "split"}` per line (FlagEmbedding-compatible).
- `results/metrics.json`, `results/train_info.json`: the evaluation metrics and training run details.
- `models/bge-base-en-v1.5-hipaa/`: the fine-tuned weights (git-ignored). See [`MODEL_CARD.md`](MODEL_CARD.md).

Downloaded regulation XML goes in `data/raw/` (git-ignored). It is only parsed, never executed. The notebook's hard-negative mining, LM-Cocktail merging and SageMaker deployment are not ported.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
