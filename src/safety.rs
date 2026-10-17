//! Conservative checks for lines that must never be hidden.

/// Returns `true` when a line is likely to be an important diagnostic.
///
/// These checks deliberately favour false positives: showing an extra line is
/// preferable to hiding a compiler error or warning.
pub fn must_show(line: &str) -> bool {
    let line = line.to_ascii_lowercase();

    [
        "warning",
        "error",
        "fatal",
        "failed",
        "failure",
        "panic",
        "traceback",
        "exception",
        "abort",
        "undefined reference",
        "cannot find",
        "no such file",
    ]
    .iter()
    .any(|needle| line.contains(needle))
}

/// Diagnostic continuations that are useful immediately after a shown warning
/// or error, such as a Rust source location or the caret below it.
pub fn is_diagnostic_continuation(line: &str) -> bool {
    let trimmed = line.trim_start();

    trimmed.starts_with("-->")
        || trimmed.starts_with('|')
        || trimmed.starts_with('^')
        || trimmed.starts_with("note:")
        || trimmed.starts_with("help:")
        || trimmed.starts_with("Caused by:")
        || trimmed.starts_with("at ")
}

#[cfg(test)]
mod tests {
    use super::{is_diagnostic_continuation, must_show};

    #[test]
    fn preserves_obvious_diagnostics() {
        assert!(must_show("warning: unused variable: `name`"));
        assert!(must_show("error[E0308]: mismatched types"));
        assert!(must_show("thread 'main' panicked at src/main.rs:8"));
    }

    #[test]
    fn identifies_diagnostic_continuations() {
        assert!(is_diagnostic_continuation("  --> src/main.rs:8:4"));
        assert!(is_diagnostic_continuation("   |     ^ expected `u8`"));
        assert!(!is_diagnostic_continuation("Compiling serde v1.0.0"));
    }
}
