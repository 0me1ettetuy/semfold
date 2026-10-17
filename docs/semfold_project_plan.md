# semfold — General Project Plan

## 1. Project Idea

`semfold` is a lightweight CLI tool that reduces noisy terminal build logs in real time.

Example:

```bash
cargo build 2>&1 | semfold
```

Instead of showing hundreds of repetitive lines, `semfold` classifies each line and collapses low-value output while keeping warnings and errors visible.

Example output:

```text
⠋ Building project... 276 routine lines hidden

⚠ warning: unused variable `x`

✖ error[E0308]: mismatched types
  --> src/main.rs:14
```

The key idea is to use a **small local ML model** so classification is fast, private, offline, and does not require an API.

---

## 2. Main AI Task

This is a **multiclass text-classification** problem.

Initial 4 classes:

```text
0 = Noise / Progress
1 = Dependency
2 = Warning
3 = Error
```

For the MVP, keep 4 classes because they are easier to train and evaluate.

---

## 3. Overall System

```text
command output
      ↓
stdin stream
      ↓
semfold Rust CLI
      ↓
small local ML model
      ↓
classify each line
      ↓
terminal renderer
      ↓
clean terminal output
```

Examples:

```bash
cargo build 2>&1 | semfold
npm install 2>&1 | semfold
docker build . 2>&1 | semfold
pytest 2>&1 | semfold
```

---

## 4. ML Workflow

```text
Existing build logs
→ real terminal/CI build output

        ↓

Extract lines
→ split each long log into individual lines

        ↓

Remap/create 4 labels
→ assign every useful line to:
   Noise/Progress, Dependency, Warning, or Error

        ↓

text,label CSV
→ create a dataset such as:

   "Compiling serde",0
   "Downloading tokio",1
   "warning: unused variable",2
   "error: build failed",3

        ↓

80/10/10 split
→ 80% training
→ 10% validation
→ 10% final testing

        ↓

Fine-tune BERT-Tiny
→ take an already pretrained small language model
   and teach it the 4 semfold categories

        ↓

Evaluate
→ Accuracy
→ F1-score
→ Warning recall
→ Error recall

        ↓

Export model.onnx
→ convert the trained model into a portable model file

        ↓

Use model inside semfold
→ Rust loads model.onnx using ONNX Runtime
```

---

## 5. Dataset

Current starting dataset: **LogChunks**.

It contains real Travis CI build logs from many open-source repositories and programming languages.

After extraction, important folders are:

```text
logchunks/LogChunks/logs/
logchunks/LogChunks/build-failure-reason/
```

Real logs are located recursively under paths similar to:

```text
logchunks/LogChunks/logs/Rust/jwilm@alacritty/failed/564527447.log
logchunks/LogChunks/logs/Python/scikit-learn@scikit-learn/failed/562398767.log
```

In Colab:

```python
import glob

log_files = glob.glob(
    "logchunks/LogChunks/logs/**/*.log",
    recursive=True
)

print("Number of logs:", len(log_files))
print(log_files[:5])
```

Inspect one log:

```python
path = log_files[0]

with open(path, "r", errors="ignore") as f:
    lines = f.read().splitlines()

print("File:", path)
print("Lines:", len(lines))
print(*lines[:30], sep="\n")
```

---

## 6. Data Preparation

Goal: transform raw logs into rows like:

```csv
text,label
"Compiling serde",0
"Downloading tokio",1
"warning: unused variable",2
"error: build failed",3
```

First approach: use simple rules to create rough labels.

```text
contains "warning"               → Warning
contains "error", "failed"       → Error
contains "download", "fetch"     → Dependency
contains "compiling", "checking" → Noise/Progress
```

Then manually inspect and correct a sample.

This is easier than manually labeling thousands of lines from zero.

---

## 7. Model

Recommended first neural model:

**BERT-Tiny (`prajjwal1/bert-tiny`)**

Why:

- very small compared with normal BERT
- already pretrained on text
- suitable for fine-tuning
- easier to run locally
- can be exported to ONNX

Conceptually:

```text
"warning: unused variable"
          ↓
      BERT-Tiny
          ↓
class probabilities
          ↓
Warning
```

You do **not** train it from scratch.

Fine-tuning means:

> Show the pretrained model many log-line examples with correct labels so it learns this specific classification task.

---

## 8. Baseline Model

Before BERT-Tiny, train a simple baseline:

**TF-IDF + Logistic Regression**

Reason:

- easy
- very fast
- gives a comparison point
- may already work surprisingly well

Compare:

```text
TF-IDF + Logistic Regression
vs.
BERT-Tiny
```

Then choose the better balance of quality and speed.

---

## 9. Metrics

Main ML metrics:

```text
Accuracy
→ how many predictions are correct overall

F1-score
→ balanced measure across classes

Warning recall
→ how many true warnings are detected

Error recall
→ how many true errors are detected
```

For `semfold`, **Error recall is especially important**.

Missing an error is much worse than showing one extra progress line.

Also measure:

```text
classification latency per line
percentage of log lines collapsed
RAM usage
```

Treat **~2 ms per line** as a performance target, not a guarantee.

---

## 10. Safety Rule

Do not rely only on ML for obvious critical errors.

Possible architecture:

```text
incoming line
     ↓
cheap safety rules
     ↓
obvious fatal/error?
     ├── yes → ALWAYS SHOW
     └── no
          ↓
      ML classifier
```

This reduces the risk of hiding an obvious critical error.

---

## 11. ONNX

ONNX is a portable model format.

Workflow:

```text
Python / PyTorch
      ↓
train BERT-Tiny
      ↓
export
      ↓
model.onnx
      ↓
Rust + ONNX Runtime
```

Training is done in Python/Colab.

The final user does **not** need Python.

---

## 12. Rust CLI

After the ML model works, build the actual `semfold` CLI in Rust.

Responsibilities:

```text
1. Read stdin line by line
2. Clean/preprocess each line
3. Pass line to local model
4. Get predicted class
5. Decide hide/show/collapse
6. Render result in terminal
```

Example:

```bash
cargo build 2>&1 | semfold
```

Possible output:

```text
⠋ Building... 57 routine lines hidden

⚠ warning: unused variable

✖ error[E0308]: mismatched types
```

---

## 13. Terminal Rendering

For the MVP, keep it simple:

```text
Noise/Progress → hide or count
Dependency     → hide or summarize
Warning        → print
Error          → print
```

Later, add ANSI terminal control codes for:

- spinner
- overwriting previous line
- live hidden-line counter
- colors

For CI/CD or non-interactive terminals, use plain-text fallback.

---

## 14. Suggested Development Order

### Phase 1 — Data

```text
1. Load LogChunks
2. Find all .log files
3. Split logs into individual lines
4. Remove empty/useless lines
5. Create rough 4-class labels
6. Save dataset.csv
7. Manually inspect/fix samples
```

### Phase 2 — ML

```text
8. Split data 80/10/10
9. Train TF-IDF + Logistic Regression baseline
10. Evaluate baseline
11. Fine-tune BERT-Tiny
12. Evaluate BERT-Tiny
13. Compare speed and quality
14. Export chosen model to ONNX
```

### Phase 3 — CLI

```text
15. Create Rust CLI
16. Read stdin line by line
17. Load ONNX model
18. Classify lines
19. Implement hide/show logic
20. Test with real commands
```

### Phase 4 — Polish

```text
21. Add ANSI folding/spinner
22. Add safety rules
23. Measure latency
24. Measure log reduction
25. Prepare final demo/results
```

---

## 15. Recommended MVP Scope

A successful course MVP:

```text
✓ real build-log dataset
✓ 4-class classifier
✓ baseline vs BERT-Tiny comparison
✓ evaluation metrics
✓ ONNX export
✓ Rust CLI reading stdin
✓ hides routine lines
✓ preserves warnings/errors
```

Optional later:

```text
- 3–5 line context window
- 5 classes
- advanced TTY redraw
- Docker/npm/cargo-specific behavior
- AUR/Flathub/COPR distribution
```

---

## 16. Main Research Question

> Can a lightweight local text-classification model reduce noisy terminal build output while preserving important warnings and errors?

---

## 17. Main Value

For developers:

- less terminal noise
- easier error spotting
- less cognitive overload
- fully local/private
- works as a normal Unix pipe
- no cloud API or large LLM required

---

## 18. Current Status

Completed:

```text
✓ project idea selected
✓ assignment proposal created
✓ ML pipeline planned
✓ LogChunks downloaded
✓ LogChunks extracted in Google Colab
✓ real .log files located
```

Next immediate step:

```text
Read several real .log files
→ inspect their lines
→ create one combined DataFrame of log lines
```

After that:

```text
raw lines
→ clean
→ label
→ dataset CSV
→ baseline model
```
