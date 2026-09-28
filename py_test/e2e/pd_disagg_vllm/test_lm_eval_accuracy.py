#!/usr/bin/env python3
"""
P/D Disaggregation LM-Eval Accuracy Test for vLLM Router

This script validates that the router correctly routes requests through
prefill and decode instances while maintaining model accuracy on standard benchmarks.

Uses the LM Evaluation Harness (lm-eval) to measure accuracy on the gsm8k task.
"""

import argparse
import os
import sys

import lm_eval
import openai

# Test configuration
TASK = "gsm8k"
DEFAULT_FILTER = "exact_match,strict-match"
RTOL = 0.03  # Relative tolerance for accuracy comparison

# Model-specific expected values (from vLLM benchmarks)
EXPECTED_VALUES = {
    ("meta-llama/Llama-3.2-1B-Instruct", DEFAULT_FILTER): 0.33,
    ("Qwen/Qwen3-0.6B", "exact_match,flexible-extract"): 0.41,
    ("deepseek-ai/deepseek-vl2-small", DEFAULT_FILTER): 0.59,
    ("deepseek-ai/deepseek-vl2-tiny", DEFAULT_FILTER): 0.19,
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", DEFAULT_FILTER): 0.65,
}

# Simple prompt for connectivity test
SIMPLE_PROMPT = (
    "The best part about working on vLLM is that I got to meet so many people across "
    "various different organizations like UCB, Google, and Meta which means"
)


class Colors:
    """Terminal colors for output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")


def run_simple_prompt(
    base_url: str,
    model_name: str,
    disable_thinking: bool = False,
) -> bool:
    """
    Run a simple prompt to verify connectivity before running full evaluation.

    Args:
        base_url: Base URL for the router API
        model_name: Model name to test

    Returns:
        True if successful, False otherwise
    """
    print_info("Running connectivity test with simple prompt...")

    try:
        # Use a very long timeout (10 minutes) for connectivity test
        client = openai.OpenAI(api_key="EMPTY", base_url=base_url, timeout=600.0)
        # Use chat completions for Instruct models
        extra_body = None
        if disable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": SIMPLE_PROMPT}],
            max_tokens=50,
            temperature=0.0,
            extra_body=extra_body,
        )

        output = completion.choices[0].message.content if completion.choices else ""

        print("-" * 60)
        print_info(f"Connectivity Test Results for {model_name}:")
        print(f"Prompt: {SIMPLE_PROMPT}")
        print(f"Output: {output}")
        print("-" * 60)

        if not output or len(output.strip()) == 0:
            print_error("Connectivity test failed: Empty output")
            return False

        print_success("Connectivity test passed")
        return True

    except Exception as e:
        print_error(f"Connectivity test failed: {e}")
        return False


def run_accuracy_evaluation(
    base_url: str,
    model_name: str,
    num_concurrent: int = 20,
    max_gen_toks: int = 256,
    disable_thinking: bool = False,
    sample: bool = False,
    limit: int = 500,
    log_samples: bool = False,
) -> dict:
    """
    Run LM Evaluation Harness on gsm8k task.

    Args:
        base_url: Base URL for the router API (should be http://host:port/v1)
        model_name: Model name to evaluate
        num_concurrent: Number of concurrent requests

    Returns:
        Dictionary containing evaluation results
    """
    print_info(f"Running LM-Eval accuracy test on {TASK} task...")
    print_info("This may take several minutes...")

    # Use Chat Completions API for Instruct models (native format)
    # This fixes 422 errors from endpoint mismatch
    try:
        gen_kwargs = {
            "max_gen_toks": max_gen_toks,
            "do_sample": sample,
        }
        if disable_thinking:
            # Qwen3 defaults to thinking mode, which can consume the entire
            # short CI generation budget before emitting a final answer.
            gen_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        if sample:
            gen_kwargs.update(
                {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                }
            )
        else:
            # Accuracy gates should be reproducible. This also avoids compiling
            # vLLM's top-k/top-p sampling kernel for every dynamic batch shape.
            gen_kwargs["temperature"] = 0.0

        results = lm_eval.simple_evaluate(
            model="local-chat-completions",
            model_args={
                "model": model_name,
                "base_url": f"{base_url}/chat/completions",
                "num_concurrent": num_concurrent,
                "max_retries": 3,
                "tokenized_requests": False,
            },
            tasks=TASK,
            num_fewshot=5,
            limit=limit,
            apply_chat_template=True,  # Enable chat template for Instruct models
            fewshot_as_multiturn=True,  # Format few-shot examples as conversation turns
            gen_kwargs=gen_kwargs,
            log_samples=log_samples,
        )
        return results

    except Exception as e:
        print_error(f"LM-Eval failed: {e}")
        raise


def validate_accuracy(
    results: dict,
    model_name: str,
    filter_name: str,
) -> bool:
    """
    Validate that accuracy meets expected thresholds.

    Args:
        results: LM-Eval results dictionary
        model_name: Model name being evaluated

    Returns:
        True if accuracy is within acceptable range, False otherwise
    """
    measured_value = results["results"][TASK][filter_name]
    expected_value = EXPECTED_VALUES.get((model_name, filter_name))

    print()
    print("=" * 60)
    print_info("Accuracy Results:")
    print_info(f"  Model:              {model_name}")
    print_info(f"  Task:               {TASK}")
    print_info(f"  Metric:             {filter_name}")
    print_info(f"  Measured Accuracy:  {measured_value:.4f}")

    if expected_value is None:
        print_warning(
            f"No expected baseline found for {model_name}. " "Cannot validate accuracy."
        )
        print_info(
            "If this is the first time testing this model, "
            "you may want to add the measured value to EXPECTED_VALUES."
        )
        print("=" * 60)
        return True  # Pass if no baseline (assume correct)

    print_info(f"  Expected Accuracy:  {expected_value:.4f}")
    print_info(f"  Tolerance:          ±{RTOL:.4f}")

    lower_bound = expected_value - RTOL
    # Higher accuracy is always acceptable, so we only enforce lower bound
    minimum_threshold = 0.3

    print_info(f"  Minimum Threshold:  {minimum_threshold:.4f}")
    print_info(f"  Lower Bound:        {lower_bound:.4f}")
    print("=" * 60)
    print()

    if measured_value >= lower_bound:
        print_success(
            f"Accuracy meets requirements! "
            f"({measured_value:.4f} vs expected {expected_value:.4f})"
        )
        return True
    else:
        print_error(
            f"Accuracy below acceptable threshold! "
            f"({measured_value:.4f} vs expected {expected_value:.4f})"
        )
        print_error(
            f"Shortfall: {lower_bound - measured_value:.4f} "
            f"(minimum: {lower_bound:.4f})"
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test P/D disaggregation accuracy using LM-Eval"
    )
    parser.add_argument(
        "--router-url",
        type=str,
        required=True,
        help="URL of the router (e.g., http://localhost:8300)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use for testing (can also use TEST_MODEL env var)",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=20,
        help="Number of concurrent requests (default: 20)",
    )
    parser.add_argument(
        "--skip-connectivity",
        action="store_true",
        help="Skip initial connectivity test",
    )
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=256,
        help="Maximum generated tokens per benchmark request (default: 256)",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=false to chat templates for bounded CI runs",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use Qwen's top-k/top-p sampling settings instead of greedy decoding",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of benchmark examples to evaluate (default: 500)",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Print the first three benchmark responses and filtered values",
    )
    parser.add_argument(
        "--filter",
        choices=("exact_match,strict-match", "exact_match,flexible-extract"),
        default=DEFAULT_FILTER,
        help="LM-Eval GSM8K metric key to validate",
    )

    args = parser.parse_args()

    # Get model name from args or environment
    model_name = args.model or os.environ.get("TEST_MODEL")
    if not model_name:
        print_error("Model name must be provided via --model or TEST_MODEL env var")
        return 1

    # Construct base URL for OpenAI API
    base_url = f"{args.router_url}/v1"

    print()
    print("=" * 60)
    print_info("P/D Disaggregation LM-Eval Accuracy Test")
    print("=" * 60)
    print_info(f"Router URL:         {args.router_url}")
    print_info(f"API Base URL:       {base_url}")
    print_info(f"Model:              {model_name}")
    print_info(f"Task:               {TASK}")
    print_info(f"Concurrent Reqs:    {args.num_concurrent}")
    print_info(f"Max Generated Toks: {args.max_gen_toks}")
    print_info(f"Thinking Enabled:   {not args.disable_thinking}")
    print_info(f"Sampling Enabled:   {args.sample}")
    print_info(f"Example Limit:      {args.limit}")
    print("=" * 60)
    print()

    # Step 1: Connectivity test (optional)
    if not args.skip_connectivity:
        if not run_simple_prompt(
            base_url,
            model_name,
            disable_thinking=args.disable_thinking,
        ):
            print_error("Connectivity test failed. Aborting evaluation.")
            return 1
        print()

    # Step 2: Run LM-Eval
    try:
        results = run_accuracy_evaluation(
            base_url=base_url,
            model_name=model_name,
            num_concurrent=args.num_concurrent,
            max_gen_toks=args.max_gen_toks,
            disable_thinking=args.disable_thinking,
            sample=args.sample,
            limit=args.limit,
            log_samples=args.log_samples,
        )
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        return 1

    if args.log_samples:
        print_info("First benchmark samples:")
        for sample in results.get("samples", {}).get(TASK, [])[:3]:
            print_info(f"  Target:   {sample.get('target')}")
            print_info(f"  Response: {sample.get('resps')}")
            print_info(f"  Filtered: {sample.get('filtered_resps')}")

    # Step 3: Validate accuracy
    if not validate_accuracy(results, model_name, args.filter):
        print_error("Accuracy validation failed!")
        return 1

    print_success("All accuracy tests PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
