use std::collections::VecDeque;
use std::io::{self, Write};

use crate::safety::{is_diagnostic_continuation, must_show};

/// The four classifier labels defined by `ml/semfold_tfidf_model_info.json`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Label {
    NoiseProgress,
    Dependency,
    Warning,
    Error,
}

impl TryFrom<i64> for Label {
    type Error = anyhow::Error;

    fn try_from(value: i64) -> Result<Self, anyhow::Error> {
        match value {
            0 => Ok(Self::NoiseProgress),
            1 => Ok(Self::Dependency),
            2 => Ok(Self::Warning),
            3 => Ok(Self::Error),
            _ => Err(anyhow::anyhow!("model returned unknown label {value}")),
        }
    }
}

/// Create the text feature format used when the ONNX model was trained.
pub fn model_input(event_text: &str, context: &str) -> String {
    if context.is_empty() {
        event_text.to_owned()
    } else {
        format!("{event_text} [CONTEXT] {context}")
    }
}

/// Recent lines retained as context for the next model input.
#[derive(Debug)]
pub struct ContextBuffer {
    lines: VecDeque<String>,
    capacity: usize,
}

impl ContextBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            lines: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn text(&self) -> String {
        self.lines.iter().cloned().collect::<Vec<_>>().join("\n")
    }

    pub fn push(&mut self, line: &str) {
        if self.capacity == 0 {
            return;
        }

        if self.lines.len() == self.capacity {
            self.lines.pop_front();
        }
        self.lines.push_back(line.to_owned());
    }
}

/// Plain-text terminal renderer. It stays deliberately pipe/CI friendly.
pub struct Folder<W: Write> {
    output: W,
    hidden_noise: usize,
    hidden_dependencies: usize,
    diagnostic_active: bool,
}

impl<W: Write> Folder<W> {
    pub fn new(output: W) -> Self {
        Self {
            output,
            hidden_noise: 0,
            hidden_dependencies: 0,
            diagnostic_active: false,
        }
    }

    pub fn process(&mut self, line: &str, label: Label) -> io::Result<()> {
        let forced =
            must_show(line) || (self.diagnostic_active && is_diagnostic_continuation(line));
        let visible = forced || matches!(label, Label::Warning | Label::Error);

        if visible {
            self.flush_summary()?;
            writeln!(self.output, "{line}")?;
            self.diagnostic_active = forced || matches!(label, Label::Warning | Label::Error);
        } else {
            match label {
                Label::NoiseProgress => self.hidden_noise += 1,
                Label::Dependency => self.hidden_dependencies += 1,
                Label::Warning | Label::Error => unreachable!("visible labels handled above"),
            }
            self.diagnostic_active = false;
        }

        Ok(())
    }

    pub fn show_unfolded(&mut self, line: &str) -> io::Result<()> {
        self.flush_summary()?;
        self.diagnostic_active = false;
        writeln!(self.output, "{line}")
    }

    pub fn finish(&mut self) -> io::Result<()> {
        self.flush_summary()
    }

    fn flush_summary(&mut self) -> io::Result<()> {
        let hidden = self.hidden_noise + self.hidden_dependencies;
        if hidden == 0 {
            return Ok(());
        }

        writeln!(self.output, "… {hidden} routine lines hidden")?;
        self.hidden_noise = 0;
        self.hidden_dependencies = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ContextBuffer, Folder, Label, model_input};

    #[test]
    fn mirrors_the_training_input_format() {
        assert_eq!(model_input("Compiling semfold", ""), "Compiling semfold");
        assert_eq!(
            model_input("error: failed", "Compiling semfold"),
            "error: failed [CONTEXT] Compiling semfold"
        );
    }

    #[test]
    fn keeps_only_the_requested_context() {
        let mut context = ContextBuffer::new(2);
        context.push("one");
        context.push("two");
        context.push("three");
        assert_eq!(context.text(), "two\nthree");
    }

    #[test]
    fn folds_routine_lines_and_shows_errors() {
        let mut output = Vec::new();
        let mut folder = Folder::new(&mut output);
        folder
            .process("Compiling serde", Label::NoiseProgress)
            .unwrap();
        folder
            .process("error[E0308]: mismatched types", Label::Error)
            .unwrap();
        folder.finish().unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("1 routine lines hidden"));
        assert!(rendered.contains("error[E0308]"));
    }
}
