use std::path::Path;

use anyhow::Context;
use ort::{session::Session, value::Tensor};

use crate::folding::Label;

/// Thin adapter around the exported scikit-learn ONNX pipeline.
pub struct Classifier {
    session: Session,
}

impl Classifier {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let environment = ort::init()
            .with_name("semfold")
            .build()
            .context("could not initialize ONNX Runtime")?;
        let session = Session::builder(&environment)
            .context("could not create ONNX Runtime session")?
            .commit_from_file(path)
            .with_context(|| format!("could not load ONNX model at {}", path.display()))?;

        Ok(Self { session })
    }

    /// Classify one model-formatted event and return its semfold label.
    pub fn classify(&mut self, input: &str) -> anyhow::Result<Label> {
        let values = [input.to_owned()];
        let input = Tensor::<String>::from_string_array(([1_usize, 1], &values[..]))
            .context("could not create ONNX string input tensor")?;
        let outputs = self
            .session
            .run(ort::inputs!["text" => input])
            .context("ONNX inference failed")?;
        let (shape, labels) = outputs["label"]
            .try_extract_tensor::<i64>()
            .context("could not read model label output")?;
        let label = *labels
            .first()
            .with_context(|| format!("model returned empty label tensor with shape {shape:?}"))?;

        Label::try_from(label)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::Classifier;

    #[test]
    fn exported_model_loads_and_returns_a_known_label() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("ml")
            .join("semfold_tfidf_logreg.onnx");
        let mut classifier = Classifier::load(&model_path).unwrap();

        classifier
            .classify("warning: unused variable `value`")
            .unwrap();
    }
}
