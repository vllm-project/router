/// Tests for PromptInput enum supporting str, list[str], list[int], list[list[int]]
use vllm_router_rs::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, GenerationRequest, PromptInput,
};

#[test]
fn test_prompt_input_single_string() {
    let json = r#"{
        "model": "test-model",
        "prompt": "Hello, world!",
        "max_tokens": 100
    }"#;

    let req: CompletionRequest = serde_json::from_str(json).unwrap();
    assert!(matches!(req.prompt, PromptInput::String(_)));

    if let PromptInput::String(s) = &req.prompt {
        assert_eq!(s, "Hello, world!");
    }

    assert_eq!(req.prompt.len(), 1);
    assert!(!req.prompt.is_empty());
    assert!(!req.prompt.is_token_based());
    assert_eq!(req.prompt.extract_text_for_routing(), "Hello, world!");
}

#[test]
fn test_chat_routing_extracts_array_developer_content() {
    let request: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [
                {"role": "developer", "content": [
                    {"type": "text", "text": "Inspect before editing."}
                ]},
                {"role": "user", "content": "Fix bug A"}
            ]
        }"#,
    )
    .unwrap();

    let routing = request.extract_text_for_routing();
    assert!(routing.contains("developer:Inspect before editing."));
    assert!(routing.contains("user:Fix bug A"));
}

#[test]
fn test_chat_routing_schemas_precede_growing_history() {
    // Stable tool schemas must come before the message history so later
    // turns keep a matchable prefix: appending schemas after the growing
    // history would stop prefix matching at the first new message.
    let request: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [{"role": "system", "content": "Use tools."}],
            "tools": [
                {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
            ]
        }"#,
    )
    .unwrap();

    let routing = request.extract_full_history_routing_text();
    let tool_pos = routing.find("tool_schema:").unwrap();
    let msg_pos = routing.find("system:").unwrap();
    assert!(
        tool_pos < msg_pos,
        "tool schemas must precede message history in the routing key: {routing}"
    );
}

#[test]
fn test_chat_routing_uses_full_history_by_default() {
    let request_a: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "developer", "content": "Always inspect before editing."},
                {"role": "user", "content": "Fix bug A"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
                    }
                }
            ]
        }"#,
    )
    .unwrap();

    let request_b: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "developer", "content": "Always inspect before editing."},
                {"role": "user", "content": "Fix bug B with a different suffix"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
                    }
                }
            ]
        }"#,
    )
    .unwrap();

    let routing_a = request_a.extract_text_for_routing();
    let routing_b = request_b.extract_text_for_routing();

    assert_ne!(routing_a, routing_b);
    assert!(routing_a.contains("system:You are a coding agent."));
    assert!(routing_a.contains("developer:Always inspect before editing."));
    assert!(routing_a.contains("user:Fix bug A"));
    assert!(routing_a.contains("tool_schema:read_file:Read a file:"));
    assert!(routing_b.contains("user:Fix bug B with a different suffix"));
}

#[test]
fn test_chat_routing_sorts_tools_for_full_history() {
    let request_a: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [{"role": "system", "content": "Use tools."}],
            "tools": [
                {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
            ]
        }"#,
    )
    .unwrap();

    let request_b: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [{"role": "system", "content": "Use tools."}],
            "tools": [
                {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}
            ]
        }"#,
    )
    .unwrap();

    assert_eq!(
        request_a.extract_text_for_routing(),
        request_b.extract_text_for_routing()
    );
}

#[test]
fn test_chat_routing_falls_back_to_session_id_without_history_text() {
    let request: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [],
            "session_params": {"session_id": "session-123"}
        }"#,
    )
    .unwrap();

    // The session key carries the \x1f terminator to avoid prefix collisions.
    assert_eq!(request.extract_text_for_routing(), "session-123\u{1f}");
}

#[test]
fn test_chat_full_history_routing_includes_volatile_turns() {
    let request_a: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "developer", "content": "Always inspect before editing."},
                {"role": "user", "content": "Fix bug A"},
                {"role": "assistant", "content": "I will inspect it.", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"a.rs\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "file contents"}]}
            ],
            "tools": [
                {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object"}}}
            ]
        }"#,
    )
    .unwrap();

    let request_b: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "developer", "content": "Always inspect before editing."},
                {"role": "user", "content": "Fix bug B"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object"}}}
            ]
        }"#,
    )
    .unwrap();

    let full_a = request_a.extract_full_history_routing_text();
    let full_b = request_b.extract_full_history_routing_text();
    assert_eq!(request_a.extract_text_for_routing(), full_a);
    assert_eq!(request_b.extract_text_for_routing(), full_b);
    assert_ne!(full_a, full_b);
    assert!(full_a.contains("user:Fix bug A"));
    assert!(full_a.contains("assistant:I will inspect it."));
    assert!(full_a.contains("assistant_tool_call:read_file:{\"path\":\"a.rs\"}"));
    assert!(full_a.contains("tool:file contents"));
    assert!(full_a.contains("tool_schema:read_file:Read a file:"));
    assert!(full_b.contains("user:Fix bug B"));
}

#[test]
fn test_prompt_input_string_array() {
    let json = r#"{
        "model": "test-model",
        "prompt": ["Hello", "world", "test"],
        "max_tokens": 100
    }"#;

    let req: CompletionRequest = serde_json::from_str(json).unwrap();
    assert!(matches!(req.prompt, PromptInput::StringArray(_)));

    if let PromptInput::StringArray(arr) = &req.prompt {
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], "Hello");
        assert_eq!(arr[1], "world");
        assert_eq!(arr[2], "test");
    }

    assert_eq!(req.prompt.len(), 3);
    assert!(!req.prompt.is_empty());
    assert!(!req.prompt.is_token_based());
    assert_eq!(req.prompt.extract_text_for_routing(), "Hello world test");
}

#[test]
fn test_prompt_input_single_int_array() {
    let json = r#"{
        "model": "test-model",
        "prompt": [128000, 9906, 11, 1917, 0],
        "max_tokens": 100
    }"#;

    let req: CompletionRequest = serde_json::from_str(json).unwrap();
    assert!(matches!(req.prompt, PromptInput::IntArray(_)));

    if let PromptInput::IntArray(ids) = &req.prompt {
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[0], 128000);
        assert_eq!(ids[1], 9906);
        assert_eq!(ids[4], 0);
    }

    assert_eq!(req.prompt.len(), 1);
    assert!(!req.prompt.is_empty());
    assert!(req.prompt.is_token_based());
    assert_eq!(req.prompt.extract_text_for_routing(), "token_ids:5");
    assert_eq!(req.prompt.estimated_token_count(), 5);
}

#[test]
fn test_prompt_input_int_batch() {
    let json = r#"{
        "model": "test-model",
        "prompt": [
            [128000, 9906, 11, 1917, 0],
            [128001, 9906, 11, 1917, 1]
        ],
        "max_tokens": 100
    }"#;

    let req: CompletionRequest = serde_json::from_str(json).unwrap();
    assert!(matches!(req.prompt, PromptInput::IntBatch(_)));

    if let PromptInput::IntBatch(batches) = &req.prompt {
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 5);
        assert_eq!(batches[1].len(), 5);
        assert_eq!(batches[0][0], 128000);
        assert_eq!(batches[1][0], 128001);
    }

    assert_eq!(req.prompt.len(), 2);
    assert!(!req.prompt.is_empty());
    assert!(req.prompt.is_token_based());
    assert_eq!(
        req.prompt.extract_text_for_routing(),
        "token_ids_batch:2:10"
    );
    assert_eq!(req.prompt.estimated_token_count(), 10);
}

#[test]
fn test_prompt_input_serialization_roundtrip_string() {
    let req = CompletionRequest {
        model: Some("test-model".to_string()),
        prompt: PromptInput::String("Hello, world!".to_string()),
        suffix: None,
        max_tokens: Some(100),
        temperature: None,
        top_p: None,
        n: None,
        stream: false,
        stream_options: None,
        logprobs: None,
        echo: false,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        best_of: None,
        logit_bias: None,
        user: None,
        seed: None,
        top_k: None,
        min_p: None,
        min_tokens: None,
        repetition_penalty: None,
        regex: None,
        ebnf: None,
        json_schema: None,
        stop_token_ids: None,
        no_stop_trim: false,
        ignore_eos: false,
        skip_special_tokens: true,
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        other: serde_json::Map::new(),
    };

    let json = serde_json::to_string(&req).unwrap();
    let parsed: CompletionRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(req.model, parsed.model);
    assert_eq!(req.prompt, parsed.prompt);
}

#[test]
fn test_prompt_input_serialization_roundtrip_int_array() {
    let req = CompletionRequest {
        model: Some("test-model".to_string()),
        prompt: PromptInput::IntArray(vec![128000, 9906, 11, 1917, 0]),
        suffix: None,
        max_tokens: Some(100),
        temperature: None,
        top_p: None,
        n: None,
        stream: false,
        stream_options: None,
        logprobs: None,
        echo: false,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        best_of: None,
        logit_bias: None,
        user: None,
        seed: None,
        top_k: None,
        min_p: None,
        min_tokens: None,
        repetition_penalty: None,
        regex: None,
        ebnf: None,
        json_schema: None,
        stop_token_ids: None,
        no_stop_trim: false,
        ignore_eos: false,
        skip_special_tokens: true,
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        other: serde_json::Map::new(),
    };

    let json = serde_json::to_string(&req).unwrap();
    let parsed: CompletionRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(req.model, parsed.model);
    assert_eq!(req.prompt, parsed.prompt);
}

#[test]
fn test_prompt_input_estimated_token_count() {
    // String: estimate ~4 chars per token
    let prompt1 = PromptInput::String("Hello, world! This is a test.".to_string()); // 30 chars
    assert_eq!(prompt1.estimated_token_count(), 7); // 30 / 4 = 7

    // String array
    let prompt2 = PromptInput::StringArray(vec![
        "Hello".to_string(), // 5 chars
        "world".to_string(), // 5 chars
    ]);
    assert_eq!(prompt2.estimated_token_count(), 2); // (5 + 5) / 4 = 2

    // Int array: exact count
    let prompt3 = PromptInput::IntArray(vec![1, 2, 3, 4, 5]);
    assert_eq!(prompt3.estimated_token_count(), 5);

    // Int batch: sum of all arrays
    let prompt4 = PromptInput::IntBatch(vec![vec![1, 2, 3], vec![4, 5, 6, 7]]);
    assert_eq!(prompt4.estimated_token_count(), 7);
}

#[test]
fn test_prompt_input_empty() {
    let prompt1 = PromptInput::String("".to_string());
    assert!(prompt1.is_empty());

    let prompt2 = PromptInput::StringArray(vec![]);
    assert!(prompt2.is_empty());

    let prompt3 = PromptInput::IntArray(vec![]);
    assert!(prompt3.is_empty());

    let prompt4 = PromptInput::IntBatch(vec![]);
    assert!(prompt4.is_empty());
}

#[test]
fn test_prompt_input_deserialization_disambiguation() {
    // Test that serde correctly disambiguates between different prompt types

    // Empty array should be IntArray (not IntBatch or StringArray)
    // Actually, empty array could be any of them due to untagged enum
    // Let's test non-empty cases

    // Array with strings
    let json1 = r#"["hello", "world"]"#;
    let prompt1: PromptInput = serde_json::from_str(json1).unwrap();
    assert!(matches!(prompt1, PromptInput::StringArray(_)));

    // Array with numbers (single-level)
    let json2 = r#"[1, 2, 3]"#;
    let prompt2: PromptInput = serde_json::from_str(json2).unwrap();
    assert!(matches!(prompt2, PromptInput::IntArray(_)));

    // Array with nested arrays (batch)
    let json3 = r#"[[1, 2], [3, 4]]"#;
    let prompt3: PromptInput = serde_json::from_str(json3).unwrap();
    assert!(matches!(prompt3, PromptInput::IntBatch(_)));

    // Single string
    let json4 = r#""hello world""#;
    let prompt4: PromptInput = serde_json::from_str(json4).unwrap();
    assert!(matches!(prompt4, PromptInput::String(_)));
}
