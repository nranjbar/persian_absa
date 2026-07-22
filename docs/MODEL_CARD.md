# Model Card

## Task

Generative Persian aspect-based sentiment analysis. Input is a Persian review; output is structured JSON containing aspect, category, opinion, and sentiment fields.

## Intended use

Research on Persian ABSA, structured generation, dataset transfer, and controlled comparison of general and Persian-oriented instruction models.

## Not intended for

High-stakes automated decisions, individual profiling, or deployment without domain-specific validation. Predictions can omit aspects, over-generate targets, misread sarcasm, or transfer poorly across annotation schemes.

## Limitations

- Domain and annotation-style shift materially affect results.
- Autoregressive inference is computationally expensive.
- Persian orthographic variation and informal language complicate span matching.
- The quality and provenance of annotation files must be independently verified.
