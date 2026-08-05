"""质量报告（Quality Reports）只读扫描服务模块（中文补充说明见下）。

职责：
- 递归扫描 ``QUALITY_REPORTS_DIR`` 下的 loop 运行清单 ``manifest.json``，
  按文件系统修改时间排序，生成面向仪表盘的紧凑摘要。
- 只读：绝不执行任何 loop 脚本，也绝不相信 manifest 内容中嵌入的路径
  （所有路径只基于 manifest 所在目录 ``run_dir``，即 manifest 的父目录）。
- 汇总指标包括：命令状态、查询评估摘要、query gate、MinerU 冒烟测试、
  服务 ingest 摘要、失败用例、agent 工具/提供方统计、检索覆盖与表格
  证据用例等。

安全要点：
- ``_build_failed_cases``、``_build_agent_metrics`` 等辅助读取只使用
  ``run_dir``（manifest 的真实父目录）下的同名 JSON 文件，
  不使用 manifest 中任何 ``path`` 字段，杜绝任意文件读取。

原英文模块 docstring（内容原样保留）：
Read-only quality report scanner for loop run manifests.
Scans QUALITY_REPORTS_DIR recursively for manifest.json files, sorts by
filesystem modified time, and returns compact summaries. Never executes
loop scripts or trusts embedded paths from manifest contents.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class QualityReportsService:
    """Read-only scanner for MinerU/RAG/Agent loop run manifests.

    MinerU/RAG/Agent loop 运行清单的只读扫描器：递归发现 manifest.json、
    解析并汇总为紧凑摘要，供质量仪表盘展示。
    """

    def __init__(self, reports_dir: Path) -> None:
        """保存报告根目录的绝对路径（``resolve()`` 规范化）。"""
        self._reports_dir = reports_dir.resolve()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def collect_runs(self, *, limit: int) -> dict[str, Any]:
        """Return structured dashboard data: runs, total_manifests,
        malformed_count, valid_total.

        返回结构化仪表盘数据：runs、total_manifests、malformed_count、
        valid_total。

        Scans ALL manifests under *reports_dir* so that *malformed_count*
        and *valid_total* are accurate across the entire directory, not
        just the first *limit* entries.
        扫描 ``reports_dir`` 下**全部** manifest，因此 ``malformed_count``
        与 ``valid_total`` 是对整个目录精确统计的，而非仅限前 ``limit`` 条。

        说明：``runs`` 只返回前 ``limit`` 条（按 mtime 倒序），但统计计数
        基于全部文件。
        """
        manifests = self._discover_manifests()
        runs: list[dict[str, Any]] = []
        malformed_count = 0
        for manifest_path, _mtime in manifests:
            summary = self._build_run_summary(manifest_path)
            if summary is None:
                malformed_count += 1  # 无法解析的 manifest 计入 malformed
            else:
                runs.append(summary)
        return {
            "runs": runs[:limit],
            "total_manifests": len(manifests),
            "malformed_count": malformed_count,
            "valid_total": len(runs),
        }

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    def _discover_manifests(self) -> list[tuple[Path, float]]:
        """Find every ``manifest.json`` under *reports_dir*, returning
        ``(path, mtime)`` sorted by mtime descending.

        递归查找 ``reports_dir`` 下所有 ``manifest.json``，返回
        ``(path, mtime)`` 列表，按修改时间倒序排列。

        说明：取 ``stat().st_mtime`` 失败（如权限问题）的文件会被跳过。
        """
        if not self._reports_dir.is_dir():
            return []
        candidates: list[tuple[Path, float]] = []
        # 按路径字典序遍历，保证确定性
        for path in sorted(self._reports_dir.rglob("manifest.json")):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue  # 无法读取修改时间则跳过该文件
            candidates.append((path, mtime))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # parsing
    # ------------------------------------------------------------------

    def _read_json_object(self, path: Path) -> dict[str, Any] | None:
        """Read *path* and return a dict if the content is a JSON object,
        otherwise return ``None``.

        读取 *path*；若内容是 JSON 对象则返回 dict，否则返回 None。

        处理三类失败（均返回 None 而非抛异常）：
        - 读取失败（OSError）或编码非 UTF-8（UnicodeDecodeError）。
        - JSON 解析失败（JSONDecodeError）。
        - 解析结果不是 dict（如数组、标量）。
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    # ------------------------------------------------------------------
    # summary builders
    # ------------------------------------------------------------------

    def _build_run_summary(self, manifest_path: Path) -> dict[str, Any] | None:
        """根据单个 manifest 构建一条运行摘要；解析失败返回 None。

        摘要字段全部从 manifest 的既存字段中安全提取（类型不匹配即缺省），
        其中一些子摘要（query、failed cases、agent metrics）会进一步读取
        manifest 同目录下的同名 JSON 工件。
        """
        manifest = self._read_json_object(manifest_path)
        if manifest is None:
            return None

        # manifest 所在目录即"运行目录"，后续副产物读取都基于它
        run_dir = manifest_path.parent
        relative_path = self._relative(run_dir)
        modified_time = self._iso_mtime(manifest_path)

        return {
            "run_id": run_dir.name,  # 目录名作为运行 ID
            "relative_path": relative_path,
            "modified_time": modified_time,
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "profile": manifest.get("profile"),
            "overall_status": manifest.get("overall_status"),
            "base_url": manifest.get("base_url"),
            "benchmark": manifest.get("benchmark"),
            "command_statuses": self._build_command_statuses(manifest),
            "query_summary": self._build_query_summary(manifest, run_dir),
            "query_gate": self._build_query_gate(manifest),
            "mineru_smoke_summary": self._build_mineru_smoke_summary(manifest),
            "service_ingest_summary": self._build_service_ingest_summary(manifest),
            "failed_cases": self._build_failed_cases(run_dir),
            "agent_metrics": self._build_agent_metrics(manifest, run_dir),
        }

    def _build_command_statuses(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        """提取 manifest 中命令列表的状态摘要。

        仅接受 ``commands`` 为列表的情况；每个字典项提取 name、
        exit_code、duration_seconds，非字典项直接跳过。
        """
        commands = manifest.get("commands")
        if not isinstance(commands, list):
            return []
        return [
            {
                "name": c.get("name"),
                "exit_code": c.get("exit_code"),
                "duration_seconds": c.get("duration_seconds"),
            }
            for c in commands
            if isinstance(c, dict)
        ]

    def _build_query_summary(
        self, manifest: dict[str, Any], run_dir: Path
    ) -> dict[str, Any] | None:
        """Extract query summary from manifest, falling back to
        same-directory ``query_eval.json``.

        从 manifest 提取查询摘要；缺失时退回读取同目录 ``query_eval.json``。

        优先级：
        1. manifest 内嵌的 ``query_eval.summary``（须为 dict）。
        2. ``run_dir / "query_eval.json"`` 的 ``summary``。
        """
        qe = manifest.get("query_eval")
        if isinstance(qe, dict):
            summary = qe.get("summary")
            if isinstance(summary, dict):
                return {
                    "total": summary.get("total"),
                    "passed": summary.get("passed"),
                    "failed": summary.get("failed"),
                }
        # Fallback: read run_dir / "query_eval.json"
        # 退回方案：读取同目录 query_eval.json
        return self._read_query_eval_json_fallback(run_dir)

    def _read_query_eval_json_fallback(self, run_dir: Path) -> dict[str, Any] | None:
        """Read ``run_dir / "query_eval.json"`` and extract its ``summary``.

        读取 ``run_dir / "query_eval.json"`` 并提取其 ``summary``。

        Never uses manifest-embedded paths — only *run_dir* (the actual
        manifest parent under QUALITY_REPORTS_DIR).
        绝不使用 manifest 中嵌入的路径——只使用 ``run_dir``（即 manifest
        在 QUALITY_REPORTS_DIR 下的真实父目录），从源头上避免路径注入。
        """
        qe_path = run_dir / "query_eval.json"
        data = self._read_json_object(qe_path)
        if data is None:
            return None
        summary = data.get("summary")
        if not isinstance(summary, dict):
            return None
        return {
            "total": summary.get("total"),
            "passed": summary.get("passed"),
            "failed": summary.get("failed"),
        }

    def _build_query_gate(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        """提取查询质量门禁（gate）信息。

        仅当 manifest 内嵌 ``query_eval.gate`` 为 dict 时返回
        {enabled, passed, checks}；否则返回 None。
        """
        qe = manifest.get("query_eval")
        if not isinstance(qe, dict):
            return None
        gate = qe.get("gate")
        if not isinstance(gate, dict):
            return None
        return {
            "enabled": gate.get("enabled", False),
            "passed": gate.get("passed"),
            "checks": gate.get("checks", {}),
        }

    def _build_mineru_smoke_summary(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        """提取 MinerU 冒烟测试摘要（``mineru_smoke.summary``）。"""
        ms = manifest.get("mineru_smoke")
        if not isinstance(ms, dict):
            return None
        summary = ms.get("summary")
        if not isinstance(summary, dict):
            return None
        return {
            "status": summary.get("status"),
            "parser_mode": summary.get("parser_mode"),
            "chunks": summary.get("chunks"),
            "error": summary.get("error"),
        }

    def _build_service_ingest_summary(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        """提取服务 ingest 摘要（``service_ingest.summary``）。"""
        si = manifest.get("service_ingest")
        if not isinstance(si, dict):
            return None
        summary = si.get("summary")
        if not isinstance(summary, dict):
            return None
        return {
            "status": summary.get("status"),
            "error": summary.get("error"),
        }

    def _build_failed_cases(self, run_dir: Path) -> list[dict[str, Any]]:
        """Read same-directory ``query_attribution.json`` and return at most
        10 failed case summaries.

        读取同目录 ``query_attribution.json``，返回至多 10 条失败用例摘要。

        过滤条件：``status != "pass"`` 的用例。每条摘要提取 id、status、
        likely_stage、failure_reasons，以及缺失的答案/引用词、来源提示
        是否命中。字段缺失时给出安全缺省值。
        """
        attr_path = run_dir / "query_attribution.json"
        data = self._read_json_object(attr_path)
        if data is None:
            return []
        cases = data.get("cases")
        if not isinstance(cases, list):
            return []
        # 只保留非 pass 的失败用例
        failed = [c for c in cases if isinstance(c, dict) and c.get("status") != "pass"]
        summaries: list[dict[str, Any]] = []
        for case in failed[:10]:
            # 各嵌套对象都可能缺失，逐层做类型防护
            answer_terms = case.get("answer_expected_terms") or {}
            citation_terms = case.get("citation_expected_terms") or {}
            source_hint = case.get("citation_source_hint") or {}
            summaries.append({
                "id": case.get("id"),
                "status": case.get("status"),
                "likely_stage": case.get("likely_stage") or "unknown",
                "failure_reasons": case.get("failure_reasons") or [],
                "missing_answer_terms": answer_terms.get("missing") if isinstance(answer_terms, dict) else [],
                "missing_citation_terms": citation_terms.get("missing") if isinstance(citation_terms, dict) else [],
                "source_hint_matched": source_hint.get("matched") if isinstance(source_hint, dict) else None,
            })
        return summaries

    # ------------------------------------------------------------------
    # agent / evidence metrics
    # ------------------------------------------------------------------

    def _build_agent_metrics(
        self, manifest: dict[str, Any], run_dir: Path
    ) -> dict[str, Any] | None:
        """Build ``agent_metrics`` from same-directory artifacts:
        manifest ``query_eval.attribution_summary``, ``query_eval.json``,
        ``query_attribution.json``, and optional agent trace sidecars.

        从同目录工件构建 ``agent_metrics``：manifest 内嵌的
        ``query_eval.attribution_summary``、``query_eval.json``、
        ``query_attribution.json`` 以及可选的 agent trace 旁车文件。

        Never trusts manifest-embedded paths — only reads from *run_dir*.
        Returns ``None`` when no query/agent artifacts are available.
        绝不信任 manifest 内嵌路径——只从 ``run_dir`` 读取。
        没有任何查询/agent 工件可用时返回 None。

        汇总内容：查询计数（total/selected/completed/passed/failed）、
        失败原因/疑似阶段计数、未归因用例 ID、检索覆盖统计、表格证据
        用例，以及 agent 工具/提供方调用统计。
        """
        # ---------- query totals ----------
        # ---------- 查询计数 ----------
        query_total: int | None = None
        query_selected: int | None = None
        query_completed: int | None = None
        query_passed: int | None = None
        query_failed: int | None = None

        # 首选：run_dir / query_eval.json 的 summary
        qe_data = self._read_json_object(run_dir / "query_eval.json")
        if isinstance(qe_data, dict):
            summary = qe_data.get("summary")
            if isinstance(summary, dict):
                query_total = _safe_int(summary.get("total"))
                query_selected = _safe_int(summary.get("selected"))
                query_completed = _safe_int(summary.get("completed"))
                query_passed = _safe_int(summary.get("passed"))
                query_failed = _safe_int(summary.get("failed"))

        # Also try manifest-embedded query_eval summary as a fallback
        # 次选：manifest 内嵌 query_eval.summary（仅当上面缺失时回填）
        qe_block = manifest.get("query_eval")
        if isinstance(qe_block, dict):
            m_summary = qe_block.get("summary")
            if isinstance(m_summary, dict):
                if query_total is None:
                    query_total = _safe_int(m_summary.get("total"))
                if query_selected is None:
                    query_selected = _safe_int(m_summary.get("selected"))
                if query_completed is None:
                    query_completed = _safe_int(m_summary.get("completed"))
                if query_passed is None:
                    query_passed = _safe_int(m_summary.get("passed"))
                if query_failed is None:
                    query_failed = _safe_int(m_summary.get("failed"))

        # ---------- stage / failure counts ----------
        # ---------- 阶段/失败计数 ----------
        failure_reason_counts: dict[str, int] = {}
        likely_stage_counts: dict[str, int] = {}
        failed_likely_stage_counts: dict[str, int] = {}
        unattributed_case_ids: list[str] = []

        # First try manifest attribution_summary (already parsed by loop runner)
        # 首选 manifest 内嵌的 attribution_summary（loop runner 已解析好）
        if isinstance(qe_block, dict):
            attr_summary = qe_block.get("attribution_summary")
            if isinstance(attr_summary, dict):
                failure_reason_counts = _coerce_string_keys(
                    attr_summary.get("failure_reason_counts")
                )
                likely_stage_counts = _coerce_string_keys(
                    attr_summary.get("likely_stage_counts")
                )
                failed_likely_stage_counts = _coerce_string_keys(
                    attr_summary.get("failed_likely_stage_counts")
                )

        # Gate unattributed case ids (from manifest gate details)
        # 未归因用例 ID（取自 manifest gate 细节）
        if isinstance(qe_block, dict):
            gate = qe_block.get("gate")
            if isinstance(gate, dict) and isinstance(gate.get("unattributed_case_ids"), list):
                unattributed_case_ids = [
                    str(cid) for cid in gate["unattributed_case_ids"]
                ]

        # ---------- attribution file (query_attribution.json) ----------
        # ---------- 归因文件 ----------
        attr_data = self._read_json_object(run_dir / "query_attribution.json")

        # If manifest attribution_summary was absent, derive from the file
        # 若 manifest 中缺 attribution_summary，则从归因文件现场推导
        if not failure_reason_counts or not likely_stage_counts:
            if isinstance(attr_data, dict):
                cases = attr_data.get("cases")
                if isinstance(cases, list):

                    if not failure_reason_counts:
                        # 统计各失败原因的出现次数
                        frc: Counter[str] = Counter()
                        for case in cases:
                            if isinstance(case, dict):
                                for reason in case.get("failure_reasons") or []:
                                    frc[str(reason)] += 1
                        failure_reason_counts = dict(frc)

                    if not likely_stage_counts:
                        # 统计疑似阶段；失败用例单独计一档
                        lsc: Counter[str] = Counter()
                        flsc: Counter[str] = Counter()
                        for case in cases:
                            if isinstance(case, dict):
                                stage = str(case.get("likely_stage") or "unknown")
                                lsc[stage] += 1
                                if case.get("status") != "pass":
                                    flsc[stage] += 1
                        likely_stage_counts = dict(lsc)
                        failed_likely_stage_counts = dict(flsc)

        # Gate unattributed case ids from attribution file fallback
        # 未归因用例 ID 的文件回退：失败且阶段为空/unknown 的用例
        if not unattributed_case_ids and isinstance(attr_data, dict):
            cases = attr_data.get("cases")
            if isinstance(cases, list):
                unattributed_case_ids = [
                    str(case.get("id"))
                    for case in cases
                    if isinstance(case, dict)
                    and case.get("status") != "pass"
                    and case.get("likely_stage") in (None, "", "unknown")
                ]

        # ---------- retrieval coverage (from attribution cases) ----------
        # ---------- 检索覆盖（取自归因用例） ----------
        retrieval_coverage: dict[str, Any] | None = None
        if isinstance(attr_data, dict):
            cases = attr_data.get("cases")
            if isinstance(cases, list):
                total = len(cases)
                with_hints = 0
                hint_matched = 0
                hint_unmatched = 0
                with_citations = 0
                for case in cases:
                    if not isinstance(case, dict):
                        continue
                    # 来源提示：matched 为 True/False 分别计数
                    src = case.get("citation_source_hint")
                    if isinstance(src, dict):
                        with_hints += 1
                        if src.get("matched") is True:
                            hint_matched += 1
                        elif src.get("matched") is False:
                            hint_unmatched += 1
                    # 引用的用例计数（citation_sources 非空）
                    sources = case.get("citation_sources")
                    if isinstance(sources, list) and len(sources) > 0:
                        with_citations += 1
                retrieval_coverage = {
                    "total_cases": total,
                    "cases_with_source_hints": with_hints,
                    "source_hint_matched": hint_matched,
                    "source_hint_unmatched": hint_unmatched,
                    "cases_with_citations": with_citations,
                    "cases_without_citations": total - with_citations,
                }

        # ---------- table evidence cases (from attribution cases) ----------
        # ---------- 表格证据用例（取自归因用例） ----------
        table_evidence_cases: list[dict[str, Any]] = []
        if isinstance(attr_data, dict):
            cases = attr_data.get("cases")
            if isinstance(cases, list):
                for case in cases:
                    if not isinstance(case, dict):
                        continue
                    cid = case.get("id")
                    # 引用摘录分组：无分组则跳过该用例
                    ceg = case.get("citation_excerpt_groups")
                    if not isinstance(ceg, list) or not ceg:
                        continue
                    groups_total = len(ceg)
                    groups_matched = sum(
                        1 for g in ceg if isinstance(g, dict) and g.get("matched")
                    )
                    # 失败原因中是否提到 table（表格相关失败）
                    has_table_failure = any(
                        "table" in str(r).lower()
                        for r in (case.get("failure_reasons") or [])
                    )
                    table_evidence_cases.append({
                        "id": str(cid),
                        "table_groups_total": groups_total,
                        "table_groups_matched": groups_matched,
                        "has_table_failure": has_table_failure,
                    })

        # ---------- agent trace sidecar (optional) ----------
        # ---------- agent trace 旁车文件（可选） ----------
        agent_tool_counts: dict[str, int] = {}
        agent_provider_counts: dict[str, int] = {}

        trace_data = self._read_agent_trace_sidecar(run_dir)
        if isinstance(trace_data, dict):
            agent_tool_counts = _coerce_string_keys(
                trace_data.get("tool_counts")
            )
            agent_provider_counts = _coerce_string_keys(
                trace_data.get("provider_counts")
            )
            # If no tool counts but has individual traces/runs, try to aggregate
            # 若没有现成计数，但存在逐条 traces/runs，则现场聚合统计
            if not agent_tool_counts or not agent_provider_counts:
                runs = trace_data.get("runs") or trace_data.get("traces")
                if isinstance(runs, list):
                    tool_counter: dict[str, int] = {}
                    provider_counter: dict[str, int] = {}
                    for run_entry in runs:
                        if not isinstance(run_entry, dict):
                            continue
                        # 提供方/模型名：provider 或 model 字段取其一
                        provider = str(
                            run_entry.get("provider")
                            or run_entry.get("model")
                            or ""
                        ).strip()
                        if provider:
                            provider_counter[provider] = (
                                provider_counter.get(provider, 0) + 1
                            )
                        # 工具名列表：tool_names 或 tools
                        tools = run_entry.get("tool_names") or run_entry.get("tools") or []
                        if isinstance(tools, list):
                            for tool in tools:
                                name = str(tool).strip()
                                if name:
                                    tool_counter[name] = tool_counter.get(name, 0) + 1
                    # 仅当原计数为空时才用聚合结果填充
                    if not agent_tool_counts:
                        agent_tool_counts = tool_counter
                    if not agent_provider_counts:
                        agent_provider_counts = provider_counter

        # Only return metrics when at least one query or trace value is present
        # 至少有一个查询/agent 数值存在时才返回指标；全空则返回 None
        if (
            query_total is None
            and query_passed is None
            and query_failed is None
            and not failure_reason_counts
            and not likely_stage_counts
            and not agent_tool_counts
            and not agent_provider_counts
        ):
            return None

        return {
            "query_total": query_total,
            "query_selected": query_selected,
            "query_completed": query_completed,
            "query_passed": query_passed,
            "query_failed": query_failed,
            "failure_reason_counts": failure_reason_counts,
            "likely_stage_counts": likely_stage_counts,
            "failed_likely_stage_counts": failed_likely_stage_counts,
            "unattributed_case_ids": unattributed_case_ids,
            "retrieval_coverage": retrieval_coverage,
            "table_evidence_cases": table_evidence_cases,
            "agent_tool_counts": agent_tool_counts,
            "agent_provider_counts": agent_provider_counts,
        }

    def _read_agent_trace_sidecar(self, run_dir: Path) -> dict[str, Any] | None:
        """Read optional agent trace sidecar from *run_dir*.

        从 *run_dir* 读取可选的 agent trace 旁车文件。

        Checks ``agent_trace_summary.json`` first, then ``agent_traces.json``.
        先检查 ``agent_trace_summary.json``，再检查 ``agent_traces.json``，
        返回第一个可读取的 JSON 对象。
        """
        for name in ("agent_trace_summary.json", "agent_traces.json"):
            data = self._read_json_object(run_dir / name)
            if data is not None:
                return data
        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _relative(self, path: Path) -> str:
        """计算 *path* 相对报告根目录的 POSIX 路径字符串。

        解析后的路径不在根目录内时（如绝对路径比较失败）退回只返回
        目录名，保证永不返回空/越界路径。
        """
        try:
            rel = path.resolve().relative_to(self._reports_dir)
        except ValueError:
            return path.name
        return rel.as_posix()

    @staticmethod
    def _iso_mtime(path: Path) -> str:
        """把文件修改时间转为 ISO 8601 字符串（UTC）。

        读取 mtime 失败（如文件已被删除）时返回空字符串。
        """
        try:
            ts = path.stat().st_mtime
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except OSError:
            return ""


def _safe_int(value: Any) -> int | None:
    """Coerce *value* to int, returning ``None`` when it can't be parsed.

    把 *value* 安全转换为 int；无法解析时返回 None（不抛异常）。
    用于把 JSON 中可能是字符串或数字的字段统一为整数。
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_string_keys(raw: Any) -> dict[str, int]:
    """Convert a raw dict-like value to ``{str: int}``, skipping non-int values.

    把原始 dict 类值转换为 ``{str: int}``；值无法转 int 的键被跳过。

    用途：把 JSON 中键为任意类型、值为数字的计数对象统一为
    字符串键 + 整数值的字典，便于后续展示与聚合。
    """
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for k, v in raw.items():
        try:
            result[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return result
