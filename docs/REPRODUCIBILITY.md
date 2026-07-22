# Reproducibility Guide

## 1. Hardware

The camera-ready rerun was reported on one NVIDIA RTX 3090 GPU. A Kaggle GPU environment can also run the pipeline, although generation time and supported batch size depend on the accelerator.

## 2. Prepare the environment

```bash
pip install -r requirements.txt
```

For gated Hugging Face models, store `HF_TOKEN` as a Kaggle Secret. Do not hard-code or commit tokens.

## 3. Configure paths and modes

The main configuration object is `Config` in `src/persian_absa_pipeline.py`.

The public release is intended primarily for `train_evaluate`: load locally supplied datasets, remove exact train-test text overlap, fine-tune/evaluate models, and export aggregate tables. The source code also retains the study's annotation-preparation functions for methodological transparency, but no annotator files or annotation records are included in the repository.

## 4. Final experiment settings

- General LLaMA3 seeds: `3407, 42, 1234, 2025, 2026`.
- One epoch.
- 4-bit QLoRA; rank 16, alpha 32, dropout 0.05.
- Adapter targets in every transformer block: attention projections and MLP projections.
- Per-device batch size 2, gradient accumulation 4, effective batch size 8.
- AdamW with 8-bit optimizer states; learning rate `1e-3`; weight decay `0.01`; warmup ratio `0.10`; linear decay.
- Prompt length 256, training sequence length 512, generation length 128.
- Greedy decoding.

## 5. Leakage and evaluation

The pipeline normalizes Persian characters and removes exact normalized train-test text overlaps before evaluating fine-tuned models. Aspect extraction uses greedy one-to-one fuzzy matching with threshold 50. Aspect-sentiment F1 requires both a matched aspect and the correct sentiment.

## 6. Outputs

The pipeline writes:

- dataset statistics;
- removed-overlap records;
- raw and parsed predictions;
- per-example error files;
- long-form seed results;
- mean/standard-deviation aggregate tables;
- training runtime and adapter checkpoints.
