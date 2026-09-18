use regex::Regex;
use std::sync::LazyLock;

static CODE_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```[a-zA-Z0-9_-]*\n?.*?```").expect("invalid regex for code blocks")
});

static INLINE_CODE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`([^`]+)`").expect("invalid regex for inline code"));

static MD_LINK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[([^\]]+)\]\([^)]+\)").expect("invalid regex for markdown link")
});

static MD_IMAGE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"!\[[^\]]*\]\([^)]+\)").expect("invalid regex for markdown image")
});

static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://[^\s)]+").expect("invalid regex for urls"));

static TABLE_ROW_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*\|.*?\|\s*$").expect("invalid regex for table row"));

static HEADING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*#{1,6}\s+").expect("invalid regex for headings"));

static BULLET_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*[-*+]\s+").expect("invalid regex for bullets"));

static NUMBERED_LIST_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*\d+\.\s+").expect("invalid regex for numbered list"));

static HTML_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<[^>]+>").expect("invalid regex for html tags"));

static MULTI_WS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[ \t]+").expect("invalid regex for whitespace"));

static MULTI_NL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n{3,}").expect("invalid regex for newlines"));

/// TextSanitizer cleans coding assistant markdown output into natural spoken text.
#[derive(Debug, Clone, Default)]
pub struct TextSanitizer {
    pub summarize_code_blocks: bool,
}

impl TextSanitizer {
    #[must_use]
    pub fn new(summarize_code_blocks: bool) -> Self {
        Self {
            summarize_code_blocks,
        }
    }

    /// Sanitizes raw LLM output into clean human-pronounceable text.
    #[must_use]
    pub fn clean(&self, text: &str) -> String {
        if text.trim().is_empty() {
            return String::new();
        }

        // 1. Handle code blocks
        let without_code_blocks = if self.summarize_code_blocks {
            CODE_BLOCK_RE.replace_all(text, " [code block omitted] ")
        } else {
            CODE_BLOCK_RE.replace_all(text, " ")
        };

        // 2. Remove markdown images completely
        let without_images = MD_IMAGE_RE.replace_all(&without_code_blocks, "");

        // 3. Keep anchor text of markdown links, drop URL
        let without_links = MD_LINK_RE.replace_all(&without_images, "$1");

        // 4. Clean standalone URLs
        let without_urls = URL_RE.replace_all(&without_links, "");

        // 5. Remove markdown tables
        let without_tables = TABLE_ROW_RE.replace_all(&without_urls, "");

        // 6. Remove headings and list markers
        let without_headings = HEADING_RE.replace_all(&without_tables, "");
        let without_bullets = BULLET_RE.replace_all(&without_headings, "");
        let without_num_list = NUMBERED_LIST_RE.replace_all(&without_bullets, "");

        // 7. Strip HTML tags
        let without_html = HTML_TAG_RE.replace_all(&without_num_list, "");

        // 8. Unescape inline code: `fn main()` -> fn main()
        let without_inline_code = INLINE_CODE_RE.replace_all(&without_html, "$1");

        // 9. Strip ASCII box-drawing characters and diagram symbols
        let cleaned_chars: String = without_inline_code
            .chars()
            .filter(|&c| !is_diagram_symbol(c))
            .collect();

        // 10. Phonetic replacements for common coding syntax
        let phonetized = normalize_code_phonetics(&cleaned_chars);

        // 11. Normalize whitespace and empty lines
        let single_spaced = MULTI_WS_RE.replace_all(&phonetized, " ");
        let cleaned_lines = Regex::new(r"(?m)^\s+$")
            .expect("valid regex")
            .replace_all(&single_spaced, "");
        let normalized_nl = MULTI_NL_RE.replace_all(&cleaned_lines, "\n\n");

        normalized_nl.trim().to_string()
    }
}

/// Identifies ASCII box-drawing and decorative diagram symbols.
fn is_diagram_symbol(c: char) -> bool {
    matches!(
        c,
        '─' | '│'
            | '┌'
            | '┐'
            | '└'
            | '┘'
            | '├'
            | '┤'
            | '┬'
            | '┴'
            | '┼'
            | '═'
            | '║'
            | '╒'
            | '╓'
            | '╔'
            | '╕'
            | '╖'
            | '╗'
            | '╘'
            | '╙'
            | '╚'
            | '╛'
            | '╜'
            | '╝'
            | '╞'
            | '╟'
            | '╠'
            | '╡'
            | '╢'
            | '╣'
            | '╤'
            | '╥'
            | '╦'
            | '╧'
            | '╨'
            | '╩'
            | '╪'
            | '╫'
            | '╬'
            | '▲'
            | '▼'
            | '►'
            | '◄'
            | '■'
            | '□'
            | '●'
            | '○'
            | '◆'
            | '◇'
            | '▶'
            | '◀'
    )
}

/// Normalizes programming syntax to sound natural when spoken.
fn normalize_code_phonetics(s: &str) -> String {
    // Replace markdown bold/italic markers (* and _)
    let mut out = s.replace("**", "").replace("__", "").replace("~~", "");

    // Technical symbols
    out = out
        .replace(" != ", " does not equal ")
        .replace(" == ", " equals ")
        .replace(" >= ", " is greater than or equal to ")
        .replace(" <= ", " is less than or equal to ")
        .replace(" && ", " and ")
        .replace(" || ", " or ")
        .replace(" -> ", " returns ")
        .replace(" => ", " maps to ");

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strips_code_blocks() {
        let sanitizer = TextSanitizer::new(false);
        let input = "Here is the code:\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\nIt works fine.";
        let cleaned = sanitizer.clean(input);
        assert_eq!(cleaned, "Here is the code:\n\nIt works fine.");
    }

    #[test]
    fn test_summarizes_code_blocks_when_configured() {
        let sanitizer = TextSanitizer::new(true);
        let input = "Here is the code:\n```rust\nfn foo() {}\n```\nAll done.";
        let cleaned = sanitizer.clean(input);
        assert!(cleaned.contains("[code block omitted]"));
    }

    #[test]
    fn test_strips_markdown_formatting_and_links() {
        let sanitizer = TextSanitizer::new(false);
        let input = "### Title\nThis is **bold** and `inline_code`. Check [our docs](https://example.com/docs).";
        let cleaned = sanitizer.clean(input);
        assert_eq!(
            cleaned,
            "Title\nThis is bold and inline_code. Check our docs."
        );
    }

    #[test]
    fn test_strips_box_drawing_diagrams() {
        let sanitizer = TextSanitizer::new(false);
        let input = "Architecture:\n┌───────┐\n│ Pipe  │ ──► [Audio]\n└───────┘";
        let cleaned = sanitizer.clean(input);
        assert!(!cleaned.contains('┌'));
        assert!(!cleaned.contains('│'));
        assert!(!cleaned.contains('└'));
        assert!(cleaned.contains("Pipe"));
        assert!(cleaned.contains("[Audio]"));
    }

    #[test]
    fn test_phonetic_normalization() {
        let sanitizer = TextSanitizer::new(false);
        let input = "If a != b && c == d { return; }";
        let cleaned = sanitizer.clean(input);
        assert!(cleaned.contains("does not equal"));
        assert!(cleaned.contains("and"));
        assert!(cleaned.contains("equals"));
    }
}
