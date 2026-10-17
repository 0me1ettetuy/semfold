use std::io::{self, BufRead};
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use semfold::{
    folding::{ContextBuffer, Folder, model_input},
    model::Classifier,
};

#[derive(Debug, Parser)]
#[command(about = "Fold routine build output while preserving warnings and errors")]
struct Cli {
    /// Path to the exported semfold ONNX model.
    #[arg(long, default_value = "ml/semfold_tfidf_logreg.onnx")]
    model: PathBuf,

    /// Number of preceding log lines included as model context.
    #[arg(long, default_value_t = 2)]
    context_lines: usize,

    /// Print stdin unchanged, without loading the model.
    #[arg(long)]
    no_fold: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut folder = Folder::new(stdout.lock());

    if cli.no_fold {
        for line in stdin.lock().lines() {
            folder.show_unfolded(&line.context("could not read stdin")?)?;
        }
        return folder.finish().context("could not write output");
    }

    let mut classifier = Classifier::load(&cli.model)?;
    let mut context = ContextBuffer::new(cli.context_lines);

    for line in stdin.lock().lines() {
        let line = line.context("could not read stdin")?;
        let input = model_input(&line, &context.text());
        let label = classifier.classify(&input)?;
        folder.process(&line, label)?;
        context.push(&line);
    }

    folder.finish().context("could not write output")
}
