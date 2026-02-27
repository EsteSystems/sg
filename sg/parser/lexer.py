"""Tokenization for the .sg contract format.

Tokens:
  KEYWORD     — gene, pathway, topology, is, risk, does, takes, gives, types,
                before, after, fails when, unhealthy when, verify, within,
                feeds, steps, requires, on failure, has, for, in, step, needs
  IDENTIFIER  — bridge_create, configuration, string, etc.
  STRING      — "quoted description"
  REFERENCE   — {bridge_name}, {interfaces}
  ARROW       — ->
  NUMBER      — 1, 2, 30
  DOT         — .
  EQUALS      — =
  COLON       — :
  DASH        — - (list item prefix)
  NEWLINE     — significant newlines
  INDENT      — increase in indentation
  DEDENT      — decrease in indentation
  EOF         — end of file
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    KEYWORD = auto()
    IDENTIFIER = auto()
    STRING = auto()
    REFERENCE = auto()
    ARROW = auto()
    NUMBER = auto()
    DOT = auto()
    EQUALS = auto()
    COLON = auto()
    DASH = auto()
    QUESTION = auto()
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line})"


# Keywords that are single words
_SINGLE_KEYWORDS = {
    "gene", "pathway", "topology",
    "is", "risk",
    "does", "takes", "gives", "types",
    "before", "after", "verify", "within", "feeds",
    "steps", "requires", "has",
    "for", "in", "step", "needs",
}

# Two-word keywords (first word -> second word)
_TWO_WORD_KEYWORDS = {
    "fails": "when",
    "unhealthy": "when",
    "on": "failure",
}


def tokenize(source: str) -> list[Token]:
    """Tokenize a .sg source string into a list of tokens."""
    tokens: list[Token] = []
    lines = source.split("\n")
    indent_stack = [0]
    in_prose_block = False
    prose_indent = 0

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip()

        if not line.strip():
            continue

        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # If we're in a prose block (after "does:"), check for section exit
        if in_prose_block:
            first_word = stripped.split()[0].rstrip(":") if stripped else ""

            is_section = (
                indent <= prose_indent
                and (
                    first_word in _SINGLE_KEYWORDS
                    or first_word in _TWO_WORD_KEYWORDS
                )
            )

            if is_section:
                in_prose_block = False
                # Fall through to normal tokenization
            else:
                # Prose line — track indentation AND emit STRING
                if indent > indent_stack[-1]:
                    indent_stack.append(indent)
                    tokens.append(Token(TokenType.INDENT, "", line_num, 1))
                else:
                    while indent < indent_stack[-1]:
                        indent_stack.pop()
                        tokens.append(Token(TokenType.DEDENT, "", line_num, 1))

                tokens.append(Token(TokenType.STRING, stripped, line_num, indent + 1))
                tokens.append(Token(TokenType.NEWLINE, "\\n", line_num, len(line) + 1))
                continue

        # Emit INDENT/DEDENT tokens
        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            tokens.append(Token(TokenType.INDENT, "", line_num, 1))
        else:
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token(TokenType.DEDENT, "", line_num, 1))

        # Tokenize the stripped line
        _tokenize_line(stripped, line_num, indent, tokens)

        # Check if we just emitted "does" + ":" — enter prose mode
        if (
            len(tokens) >= 2
            and tokens[-1].type == TokenType.COLON
            and tokens[-2].type == TokenType.KEYWORD
            and tokens[-2].value == "does"
        ):
            in_prose_block = True
            prose_indent = indent

        tokens.append(Token(TokenType.NEWLINE, "\\n", line_num, len(line) + 1))

    # Close remaining indentation
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token(TokenType.DEDENT, "", len(lines) + 1, 1))

    tokens.append(Token(TokenType.EOF, "", len(lines) + 1, 1))
    return tokens


def _tokenize_line(line: str, line_num: int, base_col: int, tokens: list[Token]) -> None:
    """Tokenize a single stripped line, appending tokens to the list."""
    i = 0
    col = base_col + 1

    while i < len(line):
        ch = line[i]

        # Skip whitespace
        if ch in (" ", "\t"):
            i += 1
            col += 1
            continue

        # String literal
        if ch == '"':
            j = i + 1
            while j < len(line) and line[j] != '"':
                j += 1
            value = line[i + 1 : j]
            tokens.append(Token(TokenType.STRING, value, line_num, col))
            i = j + 1
            col = base_col + i + 1
            continue

        # Reference: {name}
        if ch == "{":
            j = line.index("}", i)
            value = line[i + 1 : j]
            tokens.append(Token(TokenType.REFERENCE, value, line_num, col))
            i = j + 1
            col = base_col + i + 1
            continue

        # Arrow: ->
        if ch == "-" and i + 1 < len(line) and line[i + 1] == ">":
            tokens.append(Token(TokenType.ARROW, "->", line_num, col))
            i += 2
            col += 2
            continue

        # Dash (list item)
        if ch == "-" and (i + 1 >= len(line) or line[i + 1] == " "):
            tokens.append(Token(TokenType.DASH, "-", line_num, col))
            i += 1
            col += 1
            continue

        # Equals
        if ch == "=":
            tokens.append(Token(TokenType.EQUALS, "=", line_num, col))
            i += 1
            col += 1
            continue

        # Colon
        if ch == ":":
            tokens.append(Token(TokenType.COLON, ":", line_num, col))
            i += 1
            col += 1
            continue

        # Dot
        if ch == ".":
            tokens.append(Token(TokenType.DOT, ".", line_num, col))
            i += 1
            col += 1
            continue

        # Question mark (optional type)
        if ch == "?":
            tokens.append(Token(TokenType.QUESTION, "?", line_num, col))
            i += 1
            col += 1
            continue

        # Number
        if ch.isdigit():
            j = i
            while j < len(line) and line[j].isdigit():
                j += 1
            value = line[i:j]
            tokens.append(Token(TokenType.NUMBER, value, line_num, col))
            i = j
            col = base_col + i + 1
            continue

        # Square brackets (array type suffix)
        if ch == "[" and i + 1 < len(line) and line[i + 1] == "]":
            # Attach to previous identifier token as array suffix
            if tokens and tokens[-1].type == TokenType.IDENTIFIER:
                tokens[-1] = Token(
                    TokenType.IDENTIFIER,
                    tokens[-1].value + "[]",
                    tokens[-1].line,
                    tokens[-1].col,
                )
            i += 2
            col += 2
            continue

        # Word (identifier or keyword)
        if ch.isalpha() or ch == "_":
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] in ("_", "-")):
                j += 1
            word = line[i:j]

            # Check for two-word keywords
            if word in _TWO_WORD_KEYWORDS:
                expected_second = _TWO_WORD_KEYWORDS[word]
                rest = line[j:].lstrip()
                if rest.startswith(expected_second):
                    combined = f"{word} {expected_second}"
                    tokens.append(Token(TokenType.KEYWORD, combined, line_num, col))
                    j = line.index(expected_second, j) + len(expected_second)
                    i = j
                    col = base_col + i + 1
                    continue

            if word in _SINGLE_KEYWORDS:
                tokens.append(Token(TokenType.KEYWORD, word, line_num, col))
            else:
                tokens.append(Token(TokenType.IDENTIFIER, word, line_num, col))

            i = j
            col = base_col + i + 1
            continue

        # Skip any other character
        i += 1
        col += 1
