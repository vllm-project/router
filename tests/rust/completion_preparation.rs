#![cfg(all(feature = "cpu", feature = "ready"))]

//! Public, offline Completion preparation vectors using the upstream HF backend.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{json, Value};
use vllm_router_rs::backend::{
    prepare_completion, prepare_completion_classified, validate_completion_tokenizer_definition,
    CompletionPreparationError,
};
use vllm_router_rs::protocols::spec::{CompletionRequest, PromptInput};
use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer};

const MODEL_VOCAB_SIZE: u32 = 16;
const DEFINITION: &str = include_str!("fixtures/tokenizer/completion_word_level.json");

fn tokenizer() -> HuggingFaceTokenizer {
    let definition: Value = serde_json::from_str(DEFINITION).unwrap();
    validate_completion_tokenizer_definition(&definition).unwrap();
    HuggingFaceTokenizer::new_hf(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/rust/fixtures/tokenizer/completion_word_level.json"),
    )
    .unwrap()
}

fn request(value: Value) -> CompletionRequest {
    serde_json::from_value(value).unwrap()
}

#[test]
fn text_defaults_to_bos_and_honors_explicit_special_token_switch() {
    let tokenizer = tokenizer();
    for (value, expected) in [
        (json!({"prompt": "hello world"}), vec![1, 3, 4]),
        (
            json!({"prompt": "hello world", "add_special_tokens": true}),
            vec![1, 3, 4],
        ),
        (
            json!({"prompt": "hello world", "add_special_tokens": false}),
            vec![3, 4],
        ),
    ] {
        let prepared =
            prepare_completion_classified(&request(value), &tokenizer, MODEL_VOCAB_SIZE).unwrap();
        assert_eq!(prepared, expected);
    }
}

#[test]
fn text_preserves_literal_added_special_tokens_and_unicode() {
    let tokenizer = tokenizer();
    for (text, expected) in [
        ("hello<added>world", vec![3, 2, 4]),
        ("<s> hello", vec![1, 3]),
        ("你好 🙂 café", vec![5, 6, 7]),
    ] {
        let prepared = prepare_completion_classified(
            &request(json!({"prompt": text, "add_special_tokens": false})),
            &tokenizer,
            MODEL_VOCAB_SIZE,
        )
        .unwrap();
        assert_eq!(prepared, expected);
    }
}

struct CountingTokenizer {
    inner: HuggingFaceTokenizer,
    encodes: AtomicUsize,
}

impl Tokenizer for CountingTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> vllm_tokenizer::Result<Vec<u32>> {
        self.encodes.fetch_add(1, Ordering::Relaxed);
        self.inner.encode(text, add_special_tokens)
    }

    fn encode_ordinary(&self, _text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
        panic!("completion preparation must preserve added-token matching")
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> vllm_tokenizer::Result<String> {
        self.inner.decode(ids, skip_special_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

#[test]
fn token_ids_are_not_reencoded_or_limited_to_tokenizer_vocabulary() {
    let tokenizer = CountingTokenizer {
        inner: tokenizer(),
        encodes: AtomicUsize::new(0),
    };
    assert!(tokenizer.id_to_token(15).is_none());
    for add_special_tokens in [false, true] {
        let prepared = prepare_completion_classified(
            &request(json!({"prompt": [0, 2, 15], "add_special_tokens": add_special_tokens})),
            &tokenizer,
            MODEL_VOCAB_SIZE,
        )
        .unwrap();
        assert_eq!(prepared, &[0, 2, 15]);
    }
    assert_eq!(tokenizer.encodes.load(Ordering::Relaxed), 0);
}

#[test]
fn negative_out_of_range_and_zero_vocabulary_are_rejected() {
    let tokenizer = tokenizer();
    for (value, vocab_size, message) in [
        (json!({"prompt": [-1]}), MODEL_VOCAB_SIZE, "negative"),
        (json!({"prompt": [16]}), MODEL_VOCAB_SIZE, "outside"),
        (json!({"prompt": [2147483647]}), MODEL_VOCAB_SIZE, "outside"),
        (json!({"prompt": "hello"}), 3, "outside"),
        (json!({"prompt": [0]}), 0, "nonzero"),
    ] {
        let error =
            prepare_completion_classified(&request(value), &tokenizer, vocab_size).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }
}

#[test]
fn empty_encoded_input_is_rejected_but_default_bos_is_retained() {
    let tokenizer = tokenizer();
    let prepared = prepare_completion_classified(
        &request(json!({"prompt": ""})),
        &tokenizer,
        MODEL_VOCAB_SIZE,
    )
    .unwrap();
    assert_eq!(prepared, &[1]);
    let mut empty_ids = request(json!({"prompt": [0]}));
    empty_ids.prompt = PromptInput::IntArray(vec![]);
    for request in [
        empty_ids,
        request(json!({"prompt": "", "add_special_tokens": false})),
    ] {
        let error =
            prepare_completion_classified(&request, &tokenizer, MODEL_VOCAB_SIZE).unwrap_err();
        assert!(error.to_string().contains("nonempty"), "{error}");
    }
}

#[test]
fn both_batch_forms_are_rejected_even_with_one_item() {
    let tokenizer = tokenizer();
    for prompt in [json!(["hello"]), json!([[3]]), json!([])] {
        let error = prepare_completion_classified(
            &request(json!({"prompt": prompt})),
            &tokenizer,
            MODEL_VOCAB_SIZE,
        )
        .unwrap_err();
        assert!(error.to_string().contains("batched"), "{error}");
    }
}

#[test]
fn unsupported_input_modifiers_fail_closed() {
    let tokenizer = tokenizer();
    for (field, value) in [
        ("lora_path", json!("adapter")),
        ("session_params", json!({})),
        ("suffix", json!("")),
        ("truncate_prompt_tokens", json!(2)),
        ("cache_salt", json!("salt")),
        ("prompt_embeds", json!([1.0])),
        ("unknown_extension", Value::Null),
    ] {
        let mut value_map = json!({"prompt": "hello"});
        value_map[field] = value;
        let error =
            prepare_completion_classified(&request(value_map), &tokenizer, MODEL_VOCAB_SIZE)
                .unwrap_err();
        assert!(error.to_string().contains(field), "{error}");
    }
}

#[test]
fn special_token_switch_requires_a_json_boolean() {
    let tokenizer = tokenizer();
    for value in [Value::Null, json!(1), json!("false"), json!([]), json!({})] {
        let error = prepare_completion_classified(
            &request(json!({"prompt": [3], "add_special_tokens": value})),
            &tokenizer,
            MODEL_VOCAB_SIZE,
        )
        .unwrap_err();
        assert!(error.to_string().contains("boolean"), "{error}");
    }
}

#[test]
fn tokenizer_definition_rejects_padding_truncation_and_malformed_formats() {
    let definition: Value = serde_json::from_str(DEFINITION).unwrap();
    for field in ["truncation", "padding"] {
        for value in [json!({}), json!(false), json!("disabled")] {
            let mut modified = definition.clone();
            modified[field] = value;
            let error = validate_completion_tokenizer_definition(&modified).unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }
    }
    for malformed in [
        Value::Null,
        json!([]),
        json!({}),
        json!({"model": {"type": "invalid"}}),
    ] {
        assert!(validate_completion_tokenizer_definition(&malformed).is_err());
    }
}

#[test]
fn text_encodes_once_with_output_options() {
    let tokenizer = CountingTokenizer {
        inner: tokenizer(),
        encodes: AtomicUsize::new(0),
    };
    let request = request(json!({
        "prompt": "hello world", "max_tokens": 12, "temperature": 0.5,
        "top_p": 0.9, "n": 2, "stream": true, "echo": true, "logprobs": 3,
        "stop": ["done"], "presence_penalty": 0.1, "frequency_penalty": 0.2,
        "best_of": 2, "logit_bias": {"3": 1.0}, "user": "tester", "seed": 42,
        "top_k": 4, "min_p": 0.1, "min_tokens": 1, "repetition_penalty": 1.1,
        "regex": ".*", "ebnf": "root ::= 'ok'", "json_schema": "{}",
        "stop_token_ids": [4], "no_stop_trim": true, "ignore_eos": true,
        "skip_special_tokens": false, "return_hidden_states": true
    }));
    let prepared = prepare_completion_classified(&request, &tokenizer, MODEL_VOCAB_SIZE).unwrap();
    assert_eq!(prepared, &[1, 3, 4]);
    assert_eq!(tokenizer.encodes.load(Ordering::Relaxed), 1);
}

#[test]
fn diagnostic_adapter_preserves_success_and_error_text() {
    let tokenizer = tokenizer();
    let valid = request(json!({"prompt": "hello world"}));
    assert_eq!(
        prepare_completion(&valid, &tokenizer, MODEL_VOCAB_SIZE).unwrap(),
        prepare_completion_classified(&valid, &tokenizer, MODEL_VOCAB_SIZE).unwrap(),
    );
    let invalid = request(json!({"prompt": [-1]}));
    assert_eq!(
        prepare_completion(&invalid, &tokenizer, MODEL_VOCAB_SIZE).unwrap_err(),
        prepare_completion_classified(&invalid, &tokenizer, MODEL_VOCAB_SIZE)
            .unwrap_err()
            .to_string(),
    );
}

#[test]
fn classified_invalid_input_takes_precedence_over_unsupported_modifiers() {
    let tokenizer = CountingTokenizer {
        inner: tokenizer(),
        encodes: AtomicUsize::new(0),
    };
    for value in [
        json!({"prompt": [-1], "suffix": "world"}),
        json!({"prompt": [16], "lora_path": "adapter"}),
        json!({"prompt": [-1], "session_params": {}}),
        json!({"prompt": [2147483647], "unknown_extension": null}),
        json!({"prompt": [[3], [-1]], "suffix": "world"}),
        json!({"prompt": [[16]], "unknown_extension": null}),
        json!({"prompt": [[3], []], "suffix": "world"}),
        json!({"prompt": ["hello"], "suffix": "world", "add_special_tokens": null}),
        json!({"prompt": "hello", "lora_path": "adapter", "add_special_tokens": 1}),
        json!({"prompt": [3], "session_params": {}, "add_special_tokens": "false"}),
        json!({"prompt": [3], "unknown_extension": null, "add_special_tokens": {}}),
    ] {
        let request = request(value);
        let error =
            prepare_completion_classified(&request, &tokenizer, MODEL_VOCAB_SIZE).unwrap_err();
        assert!(
            matches!(error, CompletionPreparationError::InvalidRequest(_)),
            "{error:?}"
        );
    }
    let mut empty = request(json!({"prompt": [0], "suffix": "world"}));
    empty.prompt = PromptInput::IntArray(Vec::new());
    assert!(matches!(
        prepare_completion_classified(&empty, &tokenizer, MODEL_VOCAB_SIZE),
        Err(CompletionPreparationError::InvalidRequest(_))
    ));
    assert_eq!(tokenizer.encodes.load(Ordering::Relaxed), 0);
}

#[test]
fn classified_unsupported_inputs_never_encode() {
    let tokenizer = CountingTokenizer {
        inner: tokenizer(),
        encodes: AtomicUsize::new(0),
    };
    for value in [
        json!({"prompt": "hello", "suffix": "world"}),
        json!({"prompt": [3], "lora_path": "adapter"}),
        json!({"prompt": "hello", "session_params": {}}),
        json!({"prompt": ["hello", "world"]}),
        json!({"prompt": [[3], [4, 15]]}),
        json!({"prompt": "hello", "unknown_extension": null}),
    ] {
        let request = request(value);
        let error =
            prepare_completion_classified(&request, &tokenizer, MODEL_VOCAB_SIZE).unwrap_err();
        assert!(
            matches!(error, CompletionPreparationError::Unsupported(_)),
            "{error:?}"
        );
    }
    assert_eq!(tokenizer.encodes.load(Ordering::Relaxed), 0);
}

#[test]
fn classified_service_failure_is_not_invalid_or_unsupported() {
    let tokenizer = CountingTokenizer {
        inner: tokenizer(),
        encodes: AtomicUsize::new(0),
    };
    let error = prepare_completion_classified(
        &request(json!({"prompt": [-1], "suffix": "world"})),
        &tokenizer,
        0,
    )
    .unwrap_err();
    assert!(
        matches!(error, CompletionPreparationError::ServiceFailure(_)),
        "{error:?}"
    );
    assert_eq!(tokenizer.encodes.load(Ordering::Relaxed), 0);
    let error =
        prepare_completion_classified(&request(json!({"prompt": "hello world"})), &tokenizer, 3)
            .unwrap_err();
    assert!(
        matches!(error, CompletionPreparationError::ServiceFailure(_)),
        "{error:?}"
    );
    assert_eq!(tokenizer.encodes.load(Ordering::Relaxed), 1);
    let failing = HuggingFaceTokenizer::new_hf(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/rust/fixtures/tokenizer/completion_missing_unk.json"),
    )
    .unwrap();
    let request = request(json!({"prompt": "unknown token"}));
    let error = prepare_completion_classified(&request, &failing, MODEL_VOCAB_SIZE).unwrap_err();
    assert!(
        matches!(error, CompletionPreparationError::ServiceFailure(_)),
        "{error:?}"
    );
}

#[test]
fn classified_empty_encoded_input_is_invalid() {
    let tokenizer = tokenizer();
    let error = prepare_completion_classified(
        &request(json!({"prompt": "", "add_special_tokens": false})),
        &tokenizer,
        MODEL_VOCAB_SIZE,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CompletionPreparationError::InvalidRequest(_)
    ));
}

#[test]
fn scheduling_priority_is_exact_but_unproved_extensions_are_fallbacks() {
    let tokenizer = tokenizer();
    for priority in [i64::MIN, 0, i64::MAX] {
        let value = request(json!({"prompt": [3, 4], "priority": priority}));
        assert_eq!(
            prepare_completion_classified(&value, &tokenizer, MODEL_VOCAB_SIZE).unwrap(),
            &[3, 4]
        );
    }
    for (field, value) in [
        ("priority", json!("1")),
        ("watermarking", json!(true)),
        ("documents", json!([{"title": "context"}])),
        ("lora_name", json!("adapter")),
        ("cache_salt", json!("tenant")),
    ] {
        let mut value_map = json!({"prompt": [3, 4]});
        value_map[field] = value;
        assert!(matches!(
            prepare_completion_classified(&request(value_map), &tokenizer, MODEL_VOCAB_SIZE),
            Err(CompletionPreparationError::Unsupported(_))
        ));
    }
}
