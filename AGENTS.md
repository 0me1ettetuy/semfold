# semfold repository guide

## Project purpose

`semfold` is an offline Rust CLI that folds routine build-log output while keeping warnings and errors visible. The trained inference artifact is `ml/semfold_tfidf_logreg.onnx`; do not replace or regenerate it unless the task explicitly covers ML retraining.

## Repository layout

- `data/` contains the raw archive, training/validation source data, and the independently reviewed evaluation set. See `data/README.md`.
- `ml/` contains the ONNX model and its input/label contract JSON.
- `docs/` contains project and implementation plans.
- `src/` and `tests/` will contain the Rust CLI and its test fixtures once the crate is created.

## Implementation rules

- Preserve user-visible warnings and errors. Treat safety rules as a guardrail before ML output.
- Build model inputs exactly as specified by `ml/semfold_tfidf_model_info.json`.
- Keep the initial implementation as one Rust crate and avoid services, workspaces, or TTY-only behaviour.
- Maintain plain-text output for pipes and CI; terminal decoration must be optional.
- Add or update focused tests when changing classification, safety, or folding behaviour.
- Do not edit large data/model artifacts as part of normal CLI work.

## Verification

After the Rust crate exists, run the narrowest relevant checks first, then `cargo fmt --check`, `cargo clippy -- -D warnings`, and `cargo test` before handoff when feasible.

## Long-running processes

If you start a dev server, watcher, preview server, Docker Compose stack, or another long-running local process, stop it before finishing unless the user explicitly asks to leave it running. If left running at the user's request, report its command, port (if known), and how to stop it.
