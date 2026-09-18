use anyhow::{Context, Result};
use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::sync::RwLock;
use tracing::debug;

const MAX_CACHE_SIZE: usize = 2000;

/// Phonemizes English text into IPA phonemes using espeak-ng with an in-memory cache.
pub struct Phonemizer {
    language: String,
    cache: RwLock<HashMap<String, String>>,
}

impl Phonemizer {
    #[must_use]
    pub fn new(language: &str) -> Self {
        Self {
            language: language.to_string(),
            cache: RwLock::new(HashMap::with_capacity(128)),
        }
    }

    /// Converts text into IPA phoneme characters with stress marks and punctuation preserved.
    /// Checks the in-memory cache first for sub-microsecond retrieval of repeated phrases.
    pub fn phonemize(&self, text: &str) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(String::new());
        }

        // Fast path: check in-memory cache
        if let Ok(read_guard) = self.cache.read()
            && let Some(cached) = read_guard.get(text)
        {
            return Ok(cached.clone());
        }

        debug!("Phonemizing text ({}) with espeak-ng", self.language);

        let output = Command::new("espeak-ng")
            .arg("-v")
            .arg(&self.language)
            .arg("-q")
            .arg("--ipa=3")
            .arg(text)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("failed to execute espeak-ng for phonemization")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("espeak-ng phonemization failed: {stderr}");
        }

        let ipa = String::from_utf8_lossy(&output.stdout).trim().to_string();

        // Populate cache (bounded size)
        if let Ok(mut write_guard) = self.cache.write() {
            if write_guard.len() >= MAX_CACHE_SIZE {
                write_guard.clear(); // Evict on overflow to prevent unbounded memory growth
            }
            write_guard.insert(text.to_string(), ipa.clone());
        }

        Ok(ipa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phonemize_english() {
        let phonemizer = Phonemizer::new("en-gb");
        let ipa1 = phonemizer
            .phonemize("Good morning, sir.")
            .expect("phonemize failed");
        assert!(!ipa1.is_empty());
        assert!(ipa1.contains('m') || ipa1.contains('ˈ'));

        // Second call should hit the cache immediately
        let ipa2 = phonemizer
            .phonemize("Good morning, sir.")
            .expect("cached phonemize failed");
        assert_eq!(ipa1, ipa2);
    }
}
