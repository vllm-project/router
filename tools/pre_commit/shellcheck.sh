#!/usr/bin/env bash
# Lint bash scripts for undefined variables, quoting, and similar bugs.
# Use a package-manager shellcheck (apt, brew, scoop). Do not download a binary.
set -euo pipefail

is_windows_exe() {
    [[ "$1" == *.exe ]]
}

is_git_bash() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) return 0 ;;
        *) return 1 ;;
    esac
}

find_native_shellcheck() {
    local cand
    cand="$(command -v shellcheck 2>/dev/null || true)"
    if [ -n "$cand" ] && ! is_windows_exe "$cand"; then
        echo "$cand"
        return 0
    fi
    return 1
}

install_hint() {
    echo "Please install shellcheck with your package manager, then re-run:"
    echo "  Debian/Ubuntu/WSL: sudo apt-get install shellcheck"
    echo "  Fedora:            sudo dnf install ShellCheck"
    echo "  macOS:             brew install shellcheck"
    echo "  Git Bash:          scoop install shellcheck"
    echo "  https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
}

SHELLCHECK_BIN=""
if SHELLCHECK_BIN="$(find_native_shellcheck)"; then
    :
elif is_git_bash && SHELLCHECK_BIN="$(command -v shellcheck.exe 2>/dev/null || true)" && [ -n "$SHELLCHECK_BIN" ]; then
    :
else
    install_hint
    exit 1
fi

# Windows autocrlf leaves CR in the worktree while the index blob stays LF.
# The commit hook runs in WSL, where core.autocrlf is not set, so shellcheck
# would report SC1017 on every line. Ignore that only when the staged blob
# has no CR. A committed CR still fails.
SHELLCHECK_ARGS=(-s bash -x)

index_has_cr() {
    git cat-file -e ":$1" 2>/dev/null || return 1
    git show ":$1" | grep -q $'\r'
}

shellcheck_args_for() {
    local f="$1"
    SHELLCHECK_FILE_ARGS=("${SHELLCHECK_ARGS[@]}")
    if grep -q $'\r' "$f" && git cat-file -e ":$f" 2>/dev/null && ! index_has_cr "$f"; then
        SHELLCHECK_FILE_ARGS+=(-e SC1017)
    fi
}

should_lint() {
    local f="${1//\\//}"
    f="${f#./}"
    case "$f" in
        *.sh) ;;
        *) return 1 ;;
    esac
    git check-ignore -q "$f" && return 1
    return 0
}

run_shellcheck() {
    local f status=0
    for f in "$@"; do
        if should_lint "$f"; then
            shellcheck_args_for "$f"
            "$SHELLCHECK_BIN" "${SHELLCHECK_FILE_ARGS[@]}" "$f" || status=1
        fi
    done
    return "$status"
}

if [ "$#" -gt 0 ]; then
    run_shellcheck "$@"
    exit
fi

status=0
while IFS= read -r -d '' f || [ -n "$f" ]; do
    git check-ignore -q "$f" && continue
    shellcheck_args_for "$f"
    "$SHELLCHECK_BIN" "${SHELLCHECK_FILE_ARGS[@]}" "$f" || status=1
done < <(find . -path ./.git -prune -o -name "*.sh" -print0)
exit "$status"
