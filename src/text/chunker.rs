use crate::text::filter::TextSanitizer;

/// Adaptive streaming sentence chunker designed for double-buffered TTS playback.
///
/// Fast-start mode: The very first chunk emits as soon as one complete sentence is ready (~2-3s speech).
/// Sustained mode: Subsequent chunks accumulate 5-10 seconds of speech (~25-50 words) for full prosodic warmth.
#[derive(Debug)]
pub struct SentenceChunker {
    sanitizer: TextSanitizer,
    buffer: String,
    is_first_chunk: bool,
    target_words_sustained: usize,
}

impl Default for SentenceChunker {
    fn default() -> Self {
        Self::new(true, 30)
    }
}

impl SentenceChunker {
    #[must_use]
    pub fn new(summarize_code_blocks: bool, target_words_sustained: usize) -> Self {
        Self {
            sanitizer: TextSanitizer::new(summarize_code_blocks),
            buffer: String::new(),
            is_first_chunk: true,
            target_words_sustained,
        }
    }

    /// Feeds a new streaming text delta from the LLM.
    /// Returns any chunks that are ready for immediate synthesis.
    pub fn feed(&mut self, delta: &str) -> Vec<String> {
        self.buffer.push_str(delta);

        let mut ready_chunks = Vec::new();

        loop {
            let split_index = if self.is_first_chunk {
                // First chunk: fast-start on first sentence ending
                find_sentence_boundary(&self.buffer, 1)
            } else {
                // Subsequent chunks: wait until target word count is reached or paragraph break
                find_sentence_boundary(&self.buffer, self.target_words_sustained)
            };

            if let Some(idx) = split_index {
                let chunk_raw = self.buffer[..idx].to_string();
                self.buffer.drain(..idx);

                let cleaned = self.sanitizer.clean(&chunk_raw);
                if !cleaned.is_empty() {
                    self.is_first_chunk = false;
                    ready_chunks.push(cleaned);
                }
            } else {
                break;
            }
        }

        ready_chunks
    }

    /// Flushes any remaining text when the LLM stream completes.
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.trim().is_empty() {
            self.buffer.clear();
            self.is_first_chunk = true;
            return None;
        }

        let remaining = std::mem::take(&mut self.buffer);
        self.is_first_chunk = true;

        let cleaned = self.sanitizer.clean(&remaining);
        if cleaned.is_empty() {
            None
        } else {
            Some(cleaned)
        }
    }

    /// Resets the chunker buffer (used on interruption / barge-in).
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.is_first_chunk = true;
    }
}

/// Finds the byte index of a valid sentence boundary after meeting a minimum word threshold.
fn find_sentence_boundary(text: &str, min_words: usize) -> Option<usize> {
    let word_count = text.split_whitespace().count();
    if word_count < min_words {
        // Also check if there is a double newline (paragraph boundary)
        if word_count >= 1 {
            return text.find("\n\n").map(|pos| pos + 2);
        }
        return None;
    }

    let bytes = text.as_bytes();
    let len = bytes.len();

    let mut current_words = 0;
    let mut in_word = false;

    let mut i = 0;
    while i < len {
        let b = bytes[i];

        if b.is_ascii_whitespace() {
            if in_word {
                current_words += 1;
                in_word = false;
            }
        } else {
            in_word = true;
        }

        // Terminal punctuation followed by whitespace or EOF
        if (b == b'.' || b == b'!' || b == b'?') && current_words >= min_words {
            // Guard against common abbreviations (e.g., i.e., vs.)
            if !is_abbreviation(text, i) {
                // Ensure followed by whitespace or newline or end
                if i + 1 == len || bytes[i + 1].is_ascii_whitespace() {
                    return Some(i + 1);
                }
            }
        }

        // Paragraph break
        if b == b'\n' && i + 1 < len && bytes[i + 1] == b'\n' && current_words >= min_words {
            return Some(i + 2);
        }

        i += 1;
    }

    None
}

/// Checks if the period at byte `index` is part of a known abbreviation.
fn is_abbreviation(text: &str, index: usize) -> bool {
    let prefix = &text[..index];
    let token = prefix.split_whitespace().next_back().unwrap_or("");
    let token_lower = token.to_lowercase();

    matches!(
        token_lower.as_str(),
        "e.g" | "i.e" | "etc" | "vs" | "dr" | "mr" | "ms" | "prof" | "v1" | "v2" | "rs"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_start_first_chunk() {
        let mut chunker = SentenceChunker::new(false, 20);

        // First sentence should emit immediately
        let chunks = chunker.feed("Hello world, this is the first sentence. ");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world, this is the first sentence.");

        // Second sentence should wait for target word count (20 words)
        let chunks2 = chunker.feed("Short second sentence. ");
        assert!(chunks2.is_empty());

        // Feed more until sustained threshold is reached
        let chunks3 = chunker.feed("And here are a lot more words to ensure that the sustained threshold of twenty words is completely satisfied by this chunk. ");
        assert_eq!(chunks3.len(), 1);
    }

    #[test]
    fn test_handles_abbreviations() {
        let mut chunker = SentenceChunker::new(false, 1);
        let chunks = chunker.feed("Use e.g. this example right here. ");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("e.g. this example right here."));
    }

    #[test]
    fn test_flush_remaining() {
        let mut chunker = SentenceChunker::new(false, 30);
        let _ = chunker.feed("Incomplete sentence without final period");
        let flushed = chunker.flush();
        assert_eq!(
            flushed,
            Some("Incomplete sentence without final period".to_string())
        );
    }

    #[test]
    fn test_reset() {
        let mut chunker = SentenceChunker::new(false, 30);
        let _ = chunker.feed("Some buffered content");
        chunker.reset();
        assert_eq!(chunker.flush(), None);
    }
}
