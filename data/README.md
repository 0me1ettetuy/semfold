# Dataset preparation

We began with thousands of raw CI and build logs. The logs were cleaned and split into meaningful logical events: a compiler error and its related lines, for example, remain together as one sample. Each sample also retains nearby context.

Events use the four `semfold` labels:

- `0` — Noise / Progress
- `1` — Dependency
- `2` — Warning
- `3` — Error

Gemini labeled 8,400 logical events semantically, considering each event's meaning and context rather than only matching words such as `error` or `warning`. These labels were used for training and validation.

A separate set of 400 events was held out from training and directly reviewed. It is used only for final evaluation.

Both TF-IDF + Logistic Regression and BERT-Tiny were trained. TF-IDF performed better overall on the held-out 400-event test set, so that model was exported to ONNX for use by the Rust CLI.

## Files

- `semfold_model_labeled_train_source_8400.csv` — 8,400 semantically labeled events for training and validation.
- `semfold_direct_review_eval_400.csv` — independently reviewed held-out events for final evaluation.
- `LogChunks.zip` — original raw CI/build log source archive.
