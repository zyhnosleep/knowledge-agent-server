"""Shared normalization for scientific identifiers rendered as LaTeX/OCR text.

This module is used only on matching/selection paths.  Retrieved evidence is
kept verbatim so the answer and citation surfaces retain the source rendering.
"""

from __future__ import annotations

import re


_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:mathbb|mathrm|mathbf|mathsf|mathit|mathcal|mathscr|sf|bf|rm|tt|"
    r"textbf|textit|textsf|textrm|bm|boldsymbol|it|em|text|operatorname)\s*\{?",
    re.IGNORECASE,
)
_LATEX_GREEK_RE = re.compile(
    r"\\(?P<name>chi|alpha|beta|gamma|mu|delta|epsilon|theta|phi|psi|omega|"
    r"sigma|pi|tau|rho|lambda|xi|zeta|eta|nu|kappa|iota)(?![a-z])",
    re.IGNORECASE,
)
_UNICODE_GREEK = {
    "χ": "chi",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "μ": "mu",
    "δ": "delta",
    "θ": "theta",
    "φ": "phi",
    "ψ": "psi",
    "ω": "omega",
    "σ": "sigma",
    "π": "pi",
    "τ": "tau",
    "η": "eta",
    "ν": "nu",
    "κ": "kappa",
    "λ": "lambda",
    "ρ": "rho",
    "ξ": "xi",
    "ε": "epsilon",
    "ζ": "zeta",
    "ι": "iota",
}


def normalize_scientific_selector(
    value: str | None,
    *,
    preserve_boundaries: bool = False,
) -> str:
    """Return a compact, comparable key for scientific selector text.

    Font commands are wrappers rather than semantic content.  Removing their
    command and opening brace before converting Greek commands means all of
    these forms share one key: ``χ1``, ``\\chi _ { 1 }`` and
    ``\\mathbb { \\chi } _ { 1 }``.  The closing braces and TeX spacing are
    discarded at the final compaction step, so nested wrappers work without a
    growing list of anchor-specific regular expressions.
    """
    normalized = str(value or "").casefold()
    # Apply repeatedly for nested wrappers such as ``\text{\mathbf{\chi}}``.
    previous = None
    while normalized != previous:
        previous = normalized
        normalized = _LATEX_WRAPPER_RE.sub("", normalized)
    normalized = _LATEX_GREEK_RE.sub(
        lambda match: match.group("name").casefold(),
        normalized,
    )
    # TeX writes an indexed symbol with separators (and a font wrapper may
    # leave the wrapper's closing brace immediately before the subscript).
    # Those separators are structural, so ``chi _ { 1 }`` must remain ``chi1``
    # even when ordinary token boundaries are preserved for surrounding text.
    normalized = re.sub(
        r"\b(?P<name>chi|alpha|beta|gamma|mu|delta|epsilon|theta|phi|psi|omega|"
        r"sigma|pi|tau|rho|lambda|xi|zeta|eta|nu|kappa|iota)\s*\}?\s*[_^]\s*"
        r"\{?\s*(?P<index>\d+)\s*\}?",
        lambda match: f"{match.group('name')}{match.group('index')}",
        normalized,
        flags=re.IGNORECASE,
    )
    for source, replacement in _UNICODE_GREEK.items():
        normalized = normalized.replace(source, replacement)
    separator = " " if preserve_boundaries else ""
    return re.sub(r"[^a-z0-9]+", separator, normalized).strip()
