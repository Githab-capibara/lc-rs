//! Language names and ISO-ish codes.
//!
//! The language-ID model emits display names (`"Russian"`, `"English"`, …)
//! while the ASR routing speaks ISO 639-1 codes. This module bridges the two
//! and mirrors `src/localcaption/languages.py` of the Python implementation
//! exactly, including the region-qualified pseudo-codes used for Chinese
//! variants.

/// Every class of `speechbrain/lang-id-commonlanguage_ecapa`, mapped to the
/// code used internally.
const NAME_TO_CODE: [(&str, &str); 45] = [
    ("basque", "eu"),
    ("romansh_sursilvan", "rm"),
    ("sakha", "sah"),
    ("georgian", "ka"),
    ("greek", "el"),
    ("hakha_chin", "cnh"),
    ("ukrainian", "uk"),
    ("interlingua", "ia"),
    ("persian", "fa"),
    ("polish", "pl"),
    ("dutch", "nl"),
    ("chinese_hongkong", "zh-hk"),
    ("japanese", "ja"),
    ("portuguese", "pt"),
    ("italian", "it"),
    ("catalan", "ca"),
    ("chuvash", "cv"),
    ("swedish", "sv"),
    ("spanish", "es"),
    ("slovenian", "sl"),
    ("tamil", "ta"),
    ("breton", "br"),
    ("russian", "ru"),
    ("czech", "cs"),
    ("english", "en"),
    ("french", "fr"),
    ("tatar", "tt"),
    ("welsh", "cy"),
    ("kyrgyz", "ky"),
    ("esperanto", "eo"),
    ("kinyarwanda", "rw"),
    ("dhivehi", "dv"),
    ("turkish", "tr"),
    ("latvian", "lv"),
    ("estonian", "et"),
    ("arabic", "ar"),
    ("frisian", "fy"),
    ("mongolian", "mn"),
    ("chinese_taiwan", "zh-tw"),
    ("indonesian", "id"),
    ("maltese", "mt"),
    ("kabyle", "kab"),
    ("romanian", "ro"),
    ("chinese_china", "zh"),
    ("german", "de"),
];

/// Map a `SpeechBrain` label (`"Russian"`) to an ISO-ish code (`"ru"`).
///
/// Unknown names longer than three characters collapse to `"und"`; short
/// unknown names are passed through verbatim so custom codes keep working.
pub fn code_for_name(name: &str) -> String {
    let key = name.trim().trim_matches('\'').to_lowercase();
    for (candidate, code) in NAME_TO_CODE {
        if candidate == key {
            return code.to_string();
        }
    }
    if key.len() <= 3 {
        key
    } else {
        "und".to_string()
    }
}

/// Human-readable name for an ISO code, falling back to the code itself.
pub fn name_for_code(code: &str) -> String {
    let lower = code.to_lowercase();
    for (name, mapped) in NAME_TO_CODE {
        if mapped == lower {
            return display_name(name);
        }
    }
    match lower.as_str() {
        "und" => "Undetermined".to_string(),
        other => other.to_string(),
    }
}

/// Turn a `snake_case` label into a title-cased display name.
fn display_name(raw: &str) -> String {
    raw.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// All 45 label names in registry order.
pub fn all_label_names() -> Vec<&'static str> {
    NAME_TO_CODE.iter().map(|(name, _)| *name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_known_labels() {
        assert_eq!(code_for_name("Russian"), "ru");
        assert_eq!(code_for_name("russian"), "ru");
        assert_eq!(code_for_name(" English "), "en");
        assert_eq!(code_for_name("Chinese_China"), "zh");
        assert_eq!(code_for_name("chinese_taiwan"), "zh-tw");
    }

    #[test]
    fn unknown_long_names_become_undetermined() {
        assert_eq!(code_for_name("klingon"), "und");
        // Degenerate input stays degenerate, exactly like the Python map.
        assert_eq!(code_for_name(""), "");
        assert_eq!(code_for_name("'"), "");
    }

    #[test]
    fn short_unknown_codes_pass_through() {
        assert_eq!(code_for_name("de"), "de");
        assert_eq!(code_for_name("xx"), "xx");
    }

    #[test]
    fn reverse_lookup_uses_display_names() {
        assert_eq!(name_for_code("ru"), "Russian");
        assert_eq!(name_for_code("zh-tw"), "Chinese Taiwan");
        assert_eq!(name_for_code("und"), "Undetermined");
        assert_eq!(name_for_code("zz"), "zz");
    }

    #[test]
    fn registry_has_45_classes() {
        assert_eq!(all_label_names().len(), 45);
    }
}
