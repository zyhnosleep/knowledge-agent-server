"""_route_papers 动态正则缓存回归测试（2026-08-18 性能修复）。

cProfile 实测 claim 1140 单次检索 28.3s，其中 re._compile 56,060 次调用
13.4s：_route_papers 对 577 篇文档逐文档调用 _question_locks_document_subject
（每 alias 3 个动态正则）、_question_has_exact_alias→alias_in_text、
_selector_matches_text（每 selector 1 个动态正则），动态 pattern 总量超过
Python re 内部 512 条缓存 → 全打穿，每 claim 重复编译约 5 千次。
修复：按 alias/selector 做模块级 lru_cache 预编译（文档与 selector 静态，
跨 claim 全部命中）。
"""
from __future__ import annotations

import re

from app.services.paper_profile import alias_in_text
from app.services.search import QueryService
import app.services.search as search_mod


def _counting_compile(real_compile):
    counter = {"n": 0}

    def counting_compile(pattern, flags=0):
        counter["n"] += 1
        return real_compile(pattern, flags)

    counting_compile.counter = counter
    return counting_compile


def test_subject_lock_patterns_cached_across_cache_thrash(monkeypatch) -> None:
    """600 个不同 alias 打穿 re 内部 512 缓存后，重复 alias 不得重新编译。"""
    real_compile = re._compile
    counting = _counting_compile(real_compile)
    monkeypatch.setattr(search_mod.re, "_compile", counting)
    search_mod._subject_lock_patterns.cache_clear()

    for i in range(600):
        QueryService._question_locks_document_subject(f"alias-{i} 的论文", [f"alias-{i}"])

    counting.counter["n"] = 0
    QueryService._question_locks_document_subject("alias-0 的论文", ["alias-0"])
    assert counting.counter["n"] == 0, (
        "重复 alias 必须命中 lru_cache（修复前每次 3 次编译）"
    )


def test_selector_boundary_pattern_cached_across_cache_thrash(monkeypatch) -> None:
    """600 个不同 selector 打穿内部缓存后，重复 selector 不得重新编译。"""
    real_compile = re._compile
    counter = {"n": 0}

    def counting_compile(pattern, flags=0):
        # _selector_matches_text 内部还有常量 pattern（re.fullmatch 的
        # "[a-z][a-z0-9]*"）——只统计边界 pattern（含 selector 文本的编译），
        # 常量 pattern 走 re 内部缓存，不在修复范围内。
        if isinstance(pattern, str) and "sel0" in pattern:
            counter["n"] += 1
        return real_compile(pattern, flags)

    monkeypatch.setattr(search_mod.re, "_compile", counting_compile)
    search_mod._selector_boundary_pattern.cache_clear()

    for i in range(600):
        QueryService._selector_matches_text(f"sel{i}", "some unrelated text")

    counter["n"] = 0
    QueryService._selector_matches_text("sel0", "some unrelated text")
    assert counter["n"] == 0, (
        "重复 selector 必须命中 lru_cache（修复前每次 1 次编译）"
    )


def test_alias_in_text_pattern_cached_across_cache_thrash(monkeypatch) -> None:
    """600 个不同 alias 打穿内部缓存后，重复 alias 不得重新编译。"""
    import app.services.paper_profile as paper_profile

    real_compile = re._compile
    counting = _counting_compile(real_compile)
    monkeypatch.setattr(paper_profile.re, "_compile", counting)
    paper_profile._alias_boundary_pattern.cache_clear()

    for i in range(600):
        alias_in_text(f"alias-{i}", "some unrelated text")

    counting.counter["n"] = 0
    alias_in_text("alias-0", "some unrelated text")
    assert counting.counter["n"] == 0, (
        "重复 alias 必须命中 lru_cache（修复前每次 1 次编译）"
    )
