# Rust CLI plan

## Goal

Ship a single offline command-line binary that reads build output from standard input, classifies it with the existing TF-IDF + Logistic Regression ONNX model, hides routine output, and always preserves likely warnings and errors.

```text
command output → stdin → safety rules → ONNX classifier → fold renderer → terminal
```

Example target usage:

```bash
cargo build 2>&1 | semfold
```

## Recommended structure

Start as one Rust crate rather than a workspace or separate services. There is one runtime boundary (the CLI) and one model artifact, so extra layers would slow down the MVP without adding value.

```text
Cargo.toml
src/
  main.rs       # process setup, stdin loop, exit handling
  cli.rs        # command-line arguments and defaults
  model.rs      # ONNX Runtime session and model input/output conversion
  safety.rs     # conservative patterns that force important lines to display
  folding.rs    # context buffer, fold decisions, summaries, terminal output
tests/
  fixtures/     # representative Cargo/npm/Python log excerpts
  cli.rs        # end-to-end stdin-to-output tests
ml/
  semfold_tfidf_logreg.onnx
  semfold_tfidf_model_info.json
```

Keep the modules private at first. Extract a library crate only if another consumer genuinely needs classification outside this binary.

## Runtime choices

- Use `clap` for a small, explicit command-line interface.
- Use `ort` (ONNX Runtime for Rust) to execute the exported model locally.
- Use `anyhow` for actionable command-line errors.
- Start with standard output and no TTY/spinner dependency. This works correctly in CI and through pipes; interactive redraw can be added later.

The model contract must be treated as data, not duplicated in source:

- model path: `ml/semfold_tfidf_logreg.onnx`
- input name: `text`
- input shape: `[batch_size, 1]`
- input value: `event_text + " [CONTEXT] " + context` (omit the context suffix when empty)
- labels: `0` Noise / Progress, `1` Dependency, `2` Warning, `3` Error

At startup, allow an explicit `--model <path>` and otherwise locate a bundled or project-relative model. The exact output name and tensor type will be verified with a small model smoke test before the classifier implementation is finalized.

## Processing rules

1. Read stdin one line at a time and retain a bounded buffer of recent lines as context.
2. Construct the model input in the same format used during training.
3. Apply conservative safety rules before trusting the model. Obvious errors, panics, failures, warnings, and diagnostic continuations must always print.
4. For other lines, run the ONNX model.
5. Print warnings and errors. Fold Noise / Progress and Dependency lines into counters.
6. Print a plain-text summary whenever a visible event follows folded output and once at EOF.

The initial context window should be configurable and default to a small value (for example, the two preceding lines). Multi-line diagnostic grouping and TTY redraw are deferred until the line-streaming path is reliable.

## Milestones

### 1. Create and verify the crate

- Add `Cargo.toml`, the binary entry point, formatting/lint configuration, and a `--help` command.
- Add a model smoke test that loads the ONNX file, supplies a string tensor using the declared input contract, and records the output tensor name/type/shape.
- Fail clearly if the model is absent or cannot be initialized.

### 2. Implement safe plain-text folding

- Add CLI options for model location, context-window size, and an opt-out `--no-fold` mode.
- Implement the bounded context buffer, forced-show safety rules, predicted-label mapping, counters, and EOF summary.
- Ensure stdin errors and broken pipes return sensible exit statuses.

### 3. Test behaviour

- Add deterministic unit tests for input construction, safety rules, and fold summaries.
- Add end-to-end fixture tests covering routine Cargo output, warnings, Rust compiler errors with caret/arrow continuation lines, and unknown tool output.
- Run representative real logs from `data/` through the CLI and check that errors and warnings are never folded by the safety layer.

### 4. Measure and polish

- Measure model startup time, per-line latency, reduction rate, and memory use on the held-out evaluation data.
- Compare the CLI's classifications with `data/semfold_direct_review_eval_400.csv`.
- Only after the plain-text mode is solid, add optional color and TTY redraw behind a terminal-detection check.

## MVP acceptance criteria

- `some-build-command 2>&1 | semfold` works with no network access.
- The CLI loads the exported ONNX model and uses the training input format exactly.
- Obvious warnings and errors are printed even when the model would disagree.
- Routine and dependency output is folded and accurately summarized.
- Pipe/CI output remains readable plain text.
- Automated tests cover the main safety and folding paths.

