# Persian ABSA with Fine-Tuned LLaMA3

Reproducibility code for **“Aspect-Based Sentiment Analysis in Persian Using a Fine-Tuned LLaMA3 Model”** by Niloofar Ranjbar and Hamed Baghbani.

The project formulates Persian aspect-based sentiment analysis as structured generation. Given a Persian review, a model returns a JSON array of quadruples with the keys `aspect`, `category`, `opinion`, and `sentiment`.

## What is included

- `src/persian_absa_pipeline.py`: end-to-end data loading, training, generation, parsing, evaluation, and aggregation pipeline.
- `notebooks/`: Kaggle-ready notebooks with outputs and credentials removed.
- `docs/REPRODUCIBILITY.md`: workflow, settings, and expected input filenames.
- `results/RESULTS_SUMMARY.md`: aggregate results reported in the manuscript.
- `tools/set_repository_url.py`: helper for replacing the manuscript repository placeholder.

## Data availability

No benchmark dataset, raw review file, individual annotator assignment, completed annotation sheet, or consensus annotation file is distributed in this repository. Third-party datasets must be obtained from their original providers and remain governed by their original licenses and terms. The code accepts locally supplied files using the paths documented in `data/README.md` and `docs/REPRODUCIBILITY.md`.

## Quick start on Kaggle

1. Create a Kaggle notebook with a GPU accelerator.
2. Add the required datasets as Kaggle inputs using the filenames listed in `docs/REPRODUCIBILITY.md`.
3. Add a Hugging Face secret named `HF_TOKEN` when a gated model requires it.
4. Open `notebooks/Persian_ABSA_Full_Pipeline_v3.ipynb`.
5. Set the configuration paths and select `train_evaluate`.
6. For the paper settings, use seeds `3407, 42, 1234, 2025, 2026` for the general LLaMA3 experiments and the documented three seeds for Dorna.

## Main configuration

- General model: `unsloth/llama-3-8b-bnb-4bit`
- Fine-tuning: 4-bit QLoRA
- LoRA rank / alpha / dropout: `16 / 32 / 0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Epochs: `1`
- Effective batch size: `8`
- Learning rate: `1e-3`
- Maximum prompt / sequence / generation lengths: `256 / 512 / 128`
- Inference: greedy decoding, no sampling
- Aspect matching threshold: `50`

## Expected output schema

```json
[
  {
    "aspect": "باتری",
    "category": "عملکرد",
    "opinion": "زود خالی می‌شود",
    "sentiment": "negative"
  }
]
```

## Citation

A final bibliographic citation will be added when the article receives its volume, issue, page range, and DOI. Until then, use the metadata in `CITATION.cff`.

## Licenses

- Code: MIT License.
- External datasets, annotations, and model weights are not licensed or redistributed by this repository.
