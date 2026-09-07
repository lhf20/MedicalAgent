"""Shared normalization helpers for CLI commands and intent matching."""


EXIT_COMMANDS = frozenset({"exit", "quit", "q", "退出"})


def normalize_user_input(value: str) -> str:
    """Trim surrounding whitespace and trailing English/Chinese semicolons."""
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip(";；").rstrip()


def normalize_for_matching(value: str) -> str:
    """Return normalized, case-insensitive text for routing and command checks."""
    return normalize_user_input(value).casefold()


def is_exit_command(value: str) -> bool:
    """Whether user input is one of the supported CLI exit commands."""
    return normalize_for_matching(value) in EXIT_COMMANDS
