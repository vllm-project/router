#!/usr/bin/env python3
"""Check the CLI reference against the Rust declarations and Python parser.

Uses only the standard library; neither the Rust extension nor a GPU is needed.
"""

import argparse
import importlib.util
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def format_default(value):
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def python_options():
    path = ROOT / "py_src/vllm_router/router_args.py"
    spec = importlib.util.spec_from_file_location("cli_reference_router_args", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser()
    module.RouterArgs.add_cli_args(parser)
    result = {}
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._StoreTrueAction)):
            value = "flag"
        elif action.choices:
            value = ", ".join(action.choices)
        else:
            value = {str: "string", int: "integer", float: "float", None: "string"}[
                action.type
            ]
        if action.nargs in ("*", "+"):
            value += " [0..]" if action.nargs == "*" else " [1..]"
        if isinstance(action, argparse._AppendAction):
            value += " (repeatable)"
        default = (
            "n/a"
            if isinstance(action, argparse._HelpAction)
            else format_default(action.default)
        )
        for name in action.option_strings:
            if name.startswith("--"):
                result[name] = (value, default)
    return result


def rust_enum(source, name):
    body = source.split(f"pub enum {name} {{", 1)[1].split("\n}", 1)[0]
    values = dict(re.findall(r'#\[value\(name = "([^"]+)"\)\]\s*(\w+),', body))
    if len(values) != len(re.findall(r"^    \w+,$", body, re.MULTILINE)):
        raise ValueError(f"Could not read every {name} variant; update this checker")
    return values


def rust_options():
    source = (ROOT / "src/main.rs").read_text(encoding="utf-8")
    body = source.split("struct CliArgs {", 1)[1].split("\n}", 1)[0]
    enums = {
        "Backend": rust_enum(source, "Backend"),
        "KvConnector": rust_enum(
            (ROOT / "src/config/types.rs").read_text(encoding="utf-8"), "KvConnector"
        ),
    }
    fields = re.findall(
        r"#\[arg\((.*?)\)\][^\n]*\n\s*(\w+): ([^\n]+),", body, re.DOTALL
    )
    # Fail rather than silently omit a field if the declaration syntax changes.
    if len(fields) != len(re.findall(r"^    \w+: .+,$", body, re.MULTILINE)):
        raise ValueError("Could not read every CliArgs field; update this checker")
    result = {}
    for attributes, field, field_type in fields:
        long_name = re.search(r'\blong\s*=\s*"([^"]+)"', attributes)
        name = "--" + (long_name[1] if long_name else field.replace("_", "-"))
        choices = re.search(r"value_parser\s*=\s*\[(.*?)\]", attributes, re.DOTALL)
        if choices:
            value = ", ".join(re.findall(r'"([^"]+)"', choices[1]))
        elif field_type in enums:
            value = ", ".join(enums[field_type])
        else:
            if field_type == "bool":
                value = "flag"
            elif field_type in ("String", "Option<String>", "Vec<String>"):
                value = "string"
            elif field_type in ("u16", "u32", "u64", "usize"):
                value = field_type
            elif field_type in ("f32", "f64"):
                value = field_type
            else:
                raise ValueError(f"Unsupported Rust CLI type for {name}: {field_type}")
        if "num_args = 0.." in attributes:
            value += " [0..]"
        if "ArgAction::Append" in attributes or field_type.startswith("Vec<"):
            value += " (repeatable)"
        default_match = re.search(
            r"default_value(?:_t)?\s*=\s*(\"[^\"]*\"|[^,\n]+)", attributes
        )
        if default_match:
            default = default_match[1].strip().strip('"')
            if "::" in default:
                variant = default.split("::")[-1]
                default = next(
                    key for key, item in enums[field_type].items() if item == variant
                )
        else:
            default = "false" if field_type == "bool" else "unset"
            if field_type.startswith("Vec<"):
                default = "[]"
        result[name] = (value, default)
        for alias in re.findall(r'\balias\s*=\s*"([^"]+)"', attributes):
            result["--" + alias] = (value, default)
    # --prefill is consumed before clap by split_prefill_args_from_others.
    result["--prefill"] = ("URL [PORT or none] (repeatable)", "[]")
    result["--help"] = ("flag", "n/a")
    result["--version"] = ("flag", "n/a")
    return result


def main():
    rust, python = rust_options(), python_options()
    documented = {}
    for line in (
        (ROOT / "docs/cli_reference.md").read_text(encoding="utf-8").splitlines()
    ):
        if not line.startswith("| `--"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) != 6:
            raise ValueError(f"Expected six table columns: {line}")
        name = cells[0]
        if name in documented:
            raise ValueError(f"Duplicate option: {name}")
        documented[name] = cells[1:5]
    errors = []
    for name in sorted(rust.keys() | python.keys() | documented.keys()):
        expected = [*rust.get(name, ("—", "—")), *python.get(name, ("—", "—"))]
        actual = documented.get(name)
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    if errors:
        print("CLI reference is out of date:", *errors, sep="\n", file=sys.stderr)
        return 1
    print(f"CLI reference matches {len(rust)} Rust and {len(python)} Python options.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
