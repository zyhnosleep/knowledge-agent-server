from __future__ import annotations

import re


_ABSTRACT_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:abstract|\u6458\u8981)\s*"
    r"(?:(?::|\uff1a|-|\u2014)\s*(?P<inline>.*))?$",
    re.IGNORECASE,
)
_ATX_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_NUMBER = (
    r"(?:\d+(?:\.\d+)*|[ivxlcdm]+(?:\.[ivxlcdm]+)*|"
    r"[\u96f6\u3007\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]+)"
)
_SECTION_PREFIX = (
    rf"(?:\u7b2c{_NUMBER}[\u7ae0\u8282]\s*|"
    rf"[\uff08(]{_NUMBER}[\uff09)]\s*|"
    rf"{_NUMBER}\s*[.\uff0e\u3001)\uff09-]?\s*)?"
)
_SECTION_HEADING = re.compile(
    rf"^\s*{_SECTION_PREFIX}(?:"
    r"introduction|keywords?|background|methods?|methodology|results?|"
    r"conclusions?|\u5173\u952e\u8bcd|\u5f15\u8a00|\u80cc\u666f|\u65b9\u6cd5|"
    r"\u7ed3\u679c|\u7ed3\u8bba)"
    r"\s*(?:(?::|\uff1a)\s*.*)?$",
    re.IGNORECASE,
)


def is_section_boundary(line: str) -> bool:
    return bool(_ATX_HEADING.match(line) or _SECTION_HEADING.fullmatch(line))


def has_explicit_abstract(text: str) -> bool:
    return any(_ABSTRACT_HEADING.fullmatch(line) for line in text.splitlines())


def extract_explicit_abstract(candidates: list[str]) -> str | None:
    def abstract_part(value: str) -> tuple[bool, str]:
        lines = value.strip().splitlines()
        if not lines:
            return False, ""
        match = _ABSTRACT_HEADING.fullmatch(lines[0])
        if match is None:
            return False, ""
        body_lines: list[str] = []
        inline = (match.group("inline") or "").strip()
        if inline:
            body_lines.append(inline)
        for line in lines[1:]:
            if is_section_boundary(line):
                break
            body_lines.append(line)
        return True, "\n".join(body_lines).strip()

    for index, candidate in enumerate(candidates):
        matched, body = abstract_part(candidate)
        if not matched:
            continue
        if body:
            return body
        following: list[str] = []
        for next_candidate in candidates[index + 1 :]:
            next_value = next_candidate.strip()
            if not next_value:
                continue
            if is_section_boundary(next_value):
                break
            lines = next_value.splitlines()
            collected: list[str] = []
            for line in lines:
                if is_section_boundary(line):
                    break
                collected.append(line)
            if collected:
                following.append("\n".join(collected).strip())
            if len(collected) != len(lines):
                break
        joined = "\n\n".join(part for part in following if part).strip()
        return joined or None
    return None
