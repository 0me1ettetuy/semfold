# semfold documentation

`semfold` reduces noisy build output while preserving useful diagnostics. It runs locally as a Unix pipe and uses the trained ONNX classifier in [`../ml/`](../ml/).

## Current status

The data preparation and model-selection work is complete:

- 8,400 semantically labeled logical log events are available for training and validation.
- 400 separately reviewed events are reserved for final evaluation.
- TF-IDF + Logistic Regression outperformed BERT-Tiny on that final evaluation set and was exported to ONNX.

The next phase is the Rust CLI. Its implementation approach is documented in [Rust CLI plan](rust_cli_plan.md).

## Documents

- [General project plan](semfold_project_plan.md) — original project goals and ML workflow.
- [Rust CLI plan](rust_cli_plan.md) — proposed crate layout, delivery milestones, and acceptance criteria.
- [Dataset notes](../data/README.md) — event construction, labels, and data splits.

