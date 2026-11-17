// Protocol definitions and validation for various LLM APIs
// This module provides a structured approach to handling different API protocols

pub(crate) mod prometheus_text {
    //! Minimal Prometheus text-format parsing.
    //!
    //! The Router reads worker `/metrics` from more than one feature: scheduling
    //! observations and cache-metadata discovery. Both need the same split of a
    //! sample line into `name{labels} value`, so the parser lives here instead of
    //! being private to either caller.

    use std::collections::HashMap;

    /// Split one sample line into its series and its value.
    ///
    /// Returns `None` for a line without a trailing value field.
    pub(crate) fn split_series_value(line: &str) -> Option<(&str, &str)> {
        let split = line.rfind(char::is_whitespace)?;
        Some((line[..split].trim(), line[split..].trim()))
    }

    /// Split `name{k="v",...}` into the metric name and its labels.
    pub(crate) fn parse_series(series: &str) -> (&str, HashMap<String, String>) {
        let Some(open) = series.find('{') else {
            return (series, HashMap::new());
        };
        let name = &series[..open];
        let labels = series
            .strip_suffix('}')
            .map(|series| &series[open + 1..])
            .map(parse_labels)
            .unwrap_or_default();
        (name, labels)
    }

    /// Visit every sample in a text-format scrape, skipping comments and blanks.
    pub(crate) fn for_each_sample<'a>(
        text: &'a str,
        mut visit: impl FnMut(&'a str, &HashMap<String, String>, &'a str),
    ) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((series, value)) = split_series_value(line) else {
                continue;
            };
            let (name, labels) = parse_series(series);
            visit(name, &labels, value);
        }
    }

    fn parse_labels(labels: &str) -> HashMap<String, String> {
        let mut result = HashMap::new();
        let mut fields = Vec::new();
        let mut start = 0;
        let mut quoted = false;
        let mut escaped = false;
        for (index, character) in labels.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            match character {
                '\\' if quoted => escaped = true,
                '"' => quoted = !quoted,
                ',' if !quoted => {
                    fields.push(&labels[start..index]);
                    start = index + 1;
                }
                _ => {}
            }
        }
        fields.push(&labels[start..]);
        for field in fields {
            let Some((key, value)) = field.split_once('=') else {
                continue;
            };
            let value = value.trim().trim_matches('"').replace("\\\"", "\"");
            result.insert(key.trim().to_string(), value);
        }
        result
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    mod tests {
        use super::*;

        #[test]
        fn splits_series_and_value() {
            let (series, value) =
                split_series_value("vllm:cache_config_info{a=\"b\"} 1.0").unwrap();
            assert_eq!(series, "vllm:cache_config_info{a=\"b\"}");
            assert_eq!(value, "1.0");
            assert!(split_series_value("no-value").is_none());
        }

        #[test]
        fn parses_labels_with_commas_and_escapes() {
            let (name, labels) = parse_series(r#"m{a="x,y",b="say \"hi\" now",c=}"#);
            assert_eq!(name, "m");
            assert_eq!(labels["a"], "x,y");
            assert_eq!(labels["b"], "say \"hi\" now");
            assert_eq!(labels["c"], "");
        }

        #[test]
        fn visits_samples_only() {
            let text = "# HELP m help\n\nm{a=\"1\"} 2\nother 3\n";
            let mut seen = Vec::new();
            for_each_sample(text, |name, labels, value| {
                seen.push(format!(
                    "{name}:{}={value}",
                    labels.get("a").cloned().unwrap_or_default()
                ));
            });
            assert_eq!(seen, ["m:1=2", "other:=3"]);
        }
    }
}
pub mod spec;
pub mod validation;
pub mod worker_spec;
