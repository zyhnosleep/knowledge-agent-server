"""Summarize retrieval, output validity and cost without inventing answer grades."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(root: Path):
    rows = [json.loads(line) for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    unique = {row["id"]: row for row in rows}
    answerable = [row for row in unique.values() if row["original_answerability"] == "answerable"]
    report = {
        "completed_requests": len(rows), "expected_requests": 36,
        "unique_questions": len(unique),
        "answerable_gold_page_hits": sum(row["original_gold_page_hit"] for row in answerable),
        "answerable_question_count": len(answerable),
        "all_question_gold_page_hits": sum(row["original_gold_page_hit"] for row in unique.values()),
        "answer_accuracy": None,
        "answer_accuracy_note": "Needs adjudication against actually retrieved multi-page evidence; original single-page refusal labels must not be reused.",
        "modes": {},
    }
    lines = ["# 真实 RAG 首轮运行记录", "",
             f"已生成 {len(rows)}/36 条结果，涉及 {len(unique)}/18 道开发题。", "",
             "链路：项目 IngestionPipeline → canonical Parent/Child 分块 → Qwen3-Embedding-4B / SQLite-vec → QueryService.retrieve_evidence → 最多3个实际召回页面 → Qwen3-VL-8B。", "",
             f"原可回答题的参考页命中：{report['answerable_gold_page_hits']}/{len(answerable)}；该指标只反映召回，不等于答案正确率。", "",
             "| 模式 | 完成数 | JSON协议合格 | 引用ID合法 | 平均生成耗时 | 平均实际图片数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode in ("text", "text_image"):
        selected = [row for row in rows if row["mode"] == mode]
        generated = [row for row in selected if "generation_seconds" in row]
        details = {
            "completed": len(selected), "schema_valid": sum(row["schema_valid"] for row in selected),
            "citation_ids_valid": sum(row["citation_ids_valid"] for row in selected),
            "mean_generation_seconds": statistics.mean(row["generation_seconds"] for row in generated) if generated else None,
            "mean_image_count": statistics.mean(row["image_count"] for row in generated) if generated else None,
            "max_peak_allocated_gib": max((row["peak_allocated_bytes"] / 1024**3 for row in generated), default=None),
        }
        report["modes"][mode] = details
        seconds = f"{details['mean_generation_seconds']:.2f}s" if generated else "—"
        images = f"{details['mean_image_count']:.1f}" if generated else "—"
        lines.append(f"| {mode} | {len(selected)} | {details['schema_valid']} | {details['citation_ids_valid']} | {seconds} | {images} |")
    lines.extend(["", "尚未给出答案正确率。多页证据范围改变后，拒答题必须重新审核；引用ID合法也不代表该页支持答案。", "",
                  "这轮使用 PDF 文本层解析，保留原项目语义分块、索引与检索逻辑；未启用 MinerU 结构化图表解析、图片向量检索或微调。", "",
                  "显存统计含同时驻留的 Embedding 模型与视觉语言模型；时间为模型生成阶段，不含首次模型加载。", "",
                  "逐题证据见 retrievals.jsonl，模型原始输出见 predictions.jsonl，入库清单见 ingestion.json。"])
    (root / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(summarize(parser.parse_args().run_dir), ensure_ascii=False, indent=2))
