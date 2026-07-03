"""QA Assistant 自动化性能基准测试脚本 v2

Usage:
    python scripts/benchmark_qa.py                     # 运行全部测试
    python scripts/benchmark_qa.py --count 20          # 随机抽取20条
    python scripts/benchmark_qa.py --count 10 --seed 42  # 固定随机种子
    python scripts/benchmark_qa.py --no-rerank         # 对比无 rerank 性能
    python scripts/benchmark_qa.py --threshold 0.80    # 自定义阈值
    python scripts/benchmark_qa.py --output report.json  # 保存详细报告
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "shared", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "llm", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "cleaner", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "storage", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

from scripts.test_qa import HybridSearcher


def load_questions(test_file: str = "") -> list[dict]:
    path = Path(test_file or Path(__file__).parent.parent / "tests" / "test_questions.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def sample_questions(questions: list[dict], count: int, seed: int) -> list[dict]:
    if count >= len(questions):
        return questions
    rng = random.Random(seed)
    return rng.sample(questions, count)


def evaluate_keyword_coverage(answer_text: str, keywords: list[str]) -> dict:
    if not keywords:
        return {"coverage": 0.0, "hit": 0, "total": 0, "matched": []}
    matched = [kw for kw in keywords if kw in answer_text]
    return {
        "coverage": len(matched) / len(keywords),
        "hit": len(matched),
        "total": len(keywords),
        "matched": matched,
    }


def run_benchmark(questions: list[dict], searcher: HybridSearcher) -> list[dict]:
    results = []
    total = len(questions)
    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        expected_kw = q.get("expected_keywords", [])
        difficulty = q.get("difficulty", "medium")
        category = q.get("category", "")

        print(f"  [{idx}/{total}] {qid} ({difficulty}) {question_text[:40]}...", end=" ", flush=True)

        t_start = time.perf_counter()
        result = searcher.ask(question_text)
        t_end = time.perf_counter()

        total_ms = round((t_end - t_start) * 1000)

        # Determine the answer text for keyword evaluation
        if result["qa_hit"]:
            answer_text = result["qa_answer"]
            mode = "QA"
        elif result["rag_answer"]:
            answer_text = result["rag_answer"]
            mode = "RAG"
        else:
            answer_text = ""
            mode = "MISS"

        kw_eval = evaluate_keyword_coverage(answer_text, expected_kw)

        entry = {
            "id": qid,
            "category": category,
            "difficulty": difficulty,
            "question": question_text,
            "expected_intent": q.get("intent", ""),
            "actual_intent": result.get("intent", ""),
            "intent_confidence": result.get("intent_confidence", 0),
            "mode": mode,
            "qa_hit": result["qa_hit"],
            "qa_similarity": result["qa_similarity"],
            "qa_matched_question": result.get("qa_question", ""),
            "latency_ms": total_ms,
            "keyword_coverage": kw_eval["coverage"],
            "keyword_hit": kw_eval["hit"],
            "keyword_total": kw_eval["total"],
            "keywords_matched": kw_eval["matched"],
            "answer_length": len(answer_text),
            "rag_chunk_count": len(result.get("rag_chunks", [])),
            "answer_preview": answer_text[:120] if answer_text else "(无回答)",
        }
        results.append(entry)

        status_icon = "✓" if kw_eval["coverage"] >= 0.5 else ("~" if kw_eval["coverage"] > 0 else "✗")
        print(f"{status_icon} {mode:4s} {total_ms:5d}ms  kw={kw_eval['hit']}/{kw_eval['total']}")

    return results


def compute_statistics(results: list[dict]) -> dict:
    if not results:
        return {}

    latencies = [r["latency_ms"] for r in results]
    coverages = [r["keyword_coverage"] for r in results]

    qa_results = [r for r in results if r["mode"] == "QA"]
    rag_results = [r for r in results if r["mode"] == "RAG"]
    miss_results = [r for r in results if r["mode"] == "MISS"]

    stats = {
        "total": len(results),
        "qa_count": len(qa_results),
        "rag_count": len(rag_results),
        "miss_count": len(miss_results),
        "resolution_rate": round((len(qa_results) + len(rag_results)) / len(results) * 100, 1),
        "latency": {
            "avg_ms": round(sum(latencies) / len(latencies)),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "median_ms": sorted(latencies)[len(latencies) // 2],
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 5 else max(latencies),
        },
        "keyword_coverage": {
            "avg": round(sum(coverages) / len(coverages) * 100, 1),
            "full_match": sum(1 for c in coverages if c >= 1.0),
            "partial_match": sum(1 for c in coverages if 0 < c < 1.0),
            "no_match": sum(1 for c in coverages if c == 0),
        },
    }

    if qa_results:
        qa_lat = [r["latency_ms"] for r in qa_results]
        stats["qa_latency"] = {
            "avg_ms": round(sum(qa_lat) / len(qa_lat)),
            "min_ms": min(qa_lat),
            "max_ms": max(qa_lat),
        }
        stats["qa_avg_similarity"] = round(
            sum(r["qa_similarity"] for r in qa_results) / len(qa_results), 4
        )

    if rag_results:
        rag_lat = [r["latency_ms"] for r in rag_results]
        stats["rag_latency"] = {
            "avg_ms": round(sum(rag_lat) / len(rag_lat)),
            "min_ms": min(rag_lat),
            "max_ms": max(rag_lat),
        }
        stats["rag_avg_chunks"] = round(
            sum(r["rag_chunk_count"] for r in rag_results) / len(rag_results), 1
        )

    # Per difficulty
    for diff in ("easy", "medium", "hard"):
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            diff_cov = [r["keyword_coverage"] for r in diff_results]
            diff_lat = [r["latency_ms"] for r in diff_results]
            stats[f"difficulty_{diff}"] = {
                "count": len(diff_results),
                "avg_coverage": round(sum(diff_cov) / len(diff_cov) * 100, 1),
                "avg_latency_ms": round(sum(diff_lat) / len(diff_lat)),
                "resolved": sum(1 for r in diff_results if r["mode"] != "MISS"),
            }

    # Per category
    categories = set(r["category"] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            cat_cov = [r["keyword_coverage"] for r in cat_results]
            cat_lat = [r["latency_ms"] for r in cat_results]
            stats[f"category_{cat}"] = {
                "count": len(cat_results),
                "avg_coverage": round(sum(cat_cov) / len(cat_cov) * 100, 1),
                "avg_latency_ms": round(sum(cat_lat) / len(cat_lat)),
            }

    return stats


def print_report(results: list[dict], stats: dict, label: str = ""):
    print()
    print("=" * 72)
    print(f"  QA Assistant 基准测试报告 {label}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    print()
    print("  ┌─ 总体概览 ─────────────────────────────────────────────────┐")
    print(f"  │  测试总数:     {stats['total']:>3d}                                          │")
    print(f"  │  QA命中:       {stats['qa_count']:>3d}                                          │")
    print(f"  │  RAG回答:      {stats['rag_count']:>3d}                                          │")
    print(f"  │  未命中:       {stats['miss_count']:>3d}                                          │")
    print(f"  │  解决率:       {stats['resolution_rate']:>5.1f}%                                       │")
    print("  └────────────────────────────────────────────────────────────┘")

    lat = stats["latency"]
    print()
    print("  ┌─ 耗时统计 ─────────────────────────────────────────────────┐")
    print(f"  │  平均耗时:     {lat['avg_ms']:>5d} ms                                     │")
    print(f"  │  最小耗时:     {lat['min_ms']:>5d} ms                                     │")
    print(f"  │  最大耗时:     {lat['max_ms']:>5d} ms                                     │")
    print(f"  │  中位数:       {lat['median_ms']:>5d} ms                                     │")
    print(f"  │  P95:          {lat['p95_ms']:>5d} ms                                     │")
    print("  └────────────────────────────────────────────────────────────┘")

    if "qa_latency" in stats:
        qa_lat = stats["qa_latency"]
        print()
        print("  ┌─ QA模式耗时 ───────────────────────────────────────────────┐")
        print(f"  │  平均:         {qa_lat['avg_ms']:>5d} ms                                     │")
        print(f"  │  范围:         {qa_lat['min_ms']}~{qa_lat['max_ms']} ms                              │")
        print(f"  │  平均相似度:   {stats.get('qa_avg_similarity', 0):.4f}                                  │")
        print("  └────────────────────────────────────────────────────────────┘")

    if "rag_latency" in stats:
        rag_lat = stats["rag_latency"]
        print()
        print("  ┌─ RAG模式耗时 ──────────────────────────────────────────────┐")
        print(f"  │  平均:         {rag_lat['avg_ms']:>5d} ms                                     │")
        print(f"  │  范围:         {rag_lat['min_ms']}~{rag_lat['max_ms']} ms                              │")
        print(f"  │  平均检索片段: {stats.get('rag_avg_chunks', 0):.1f}                                       │")
        print("  └────────────────────────────────────────────────────────────┘")

    kw = stats["keyword_coverage"]
    print()
    print("  ┌─ 关键词覆盖率 ─────────────────────────────────────────────┐")
    print(f"  │  平均覆盖率:   {kw['avg']:>5.1f}%                                       │")
    print(f"  │  完全匹配:     {kw['full_match']:>3d}                                          │")
    print(f"  │  部分匹配:     {kw['partial_match']:>3d}                                          │")
    print(f"  │  无匹配:       {kw['no_match']:>3d}                                          │")
    print("  └────────────────────────────────────────────────────────────┘")

    # Per difficulty
    print()
    print("  ┌─ 按难度分析 ───────────────────────────────────────────────┐")
    print(f"  │  {'难度':<8} {'数量':>4} {'解决':>4} {'覆盖率':>8} {'平均耗时':>8}       │")
    print(f"  │  {'─'*8} {'─'*4} {'─'*4} {'─'*8} {'─'*8}       │")
    for diff in ("easy", "medium", "hard"):
        key = f"difficulty_{diff}"
        if key in stats:
            d = stats[key]
            diff_label = {"easy": "简单", "medium": "中等", "hard": "困难"}[diff]
            print(f"  │  {diff_label:<8} {d['count']:>4} {d['resolved']:>4} {d['avg_coverage']:>6.1f}% {d['avg_latency_ms']:>6d}ms       │")
    print("  └────────────────────────────────────────────────────────────┘")

    # Per category
    print()
    print("  ┌─ 按类别分析 ───────────────────────────────────────────────┐")
    print(f"  │  {'类别':<14} {'数量':>4} {'覆盖率':>8} {'平均耗时':>8}       │")
    print(f"  │  {'─'*14} {'─'*4} {'─'*8} {'─'*8}       │")
    for key, value in stats.items():
        if key.startswith("category_") and isinstance(value, dict):
            cat_name = key.replace("category_", "")
            print(f"  │  {cat_name:<14} {value['count']:>4} {value['avg_coverage']:>6.1f}% {value['avg_latency_ms']:>6d}ms       │")
    print("  └────────────────────────────────────────────────────────────┘")

    # Per-question breakdown
    print()
    print("  ┌─ 逐题详情 ─────────────────────────────────────────────────┐")
    for r in results:
        status = "✓" if r["keyword_coverage"] >= 0.5 else ("~" if r["keyword_coverage"] > 0 else "✗")
        intent_match = "✓" if r["actual_intent"] == r["expected_intent"] else "✗"
        print(f"  │ {status} {r['id']:<6} {r['mode']:4s} {r['latency_ms']:>5d}ms "
              f"kw={r['keyword_hit']}/{r['keyword_total']} "
              f"意图{intent_match}  {r['question'][:28]}")
    print("  └────────────────────────────────────────────────────────────┘")

    # Performance bottleneck hints
    print()
    print("  ┌─ 性能瓶颈分析提示 ─────────────────────────────────────────┐")
    if "qa_latency" in stats and "rag_latency" in stats:
        qa_avg = stats["qa_latency"]["avg_ms"]
        rag_avg = stats["rag_latency"]["avg_ms"]
        diff = rag_avg - qa_avg
        if diff > 1000:
            print(f"  │  ⚠ RAG比QA平均慢 {diff}ms → LLM生成阶段可能是主要瓶颈       │")
        elif diff > 500:
            print(f"  │  △ RAG比QA平均慢 {diff}ms → LLM生成有可观察延迟              │")
        else:
            print(f"  │  ✓ QA/RAG耗时差异较小 ({diff}ms)                             │")

    slow_items = sorted(results, key=lambda x: x["latency_ms"], reverse=True)[:3]
    print(f"  │  最慢3题: {', '.join(r['id'] + '(' + str(r['latency_ms']) + 'ms)' for r in slow_items)}")

    miss_items = [r for r in results if r["mode"] == "MISS"]
    if miss_items:
        print(f"  │  未命中题: {', '.join(r['id'] for r in miss_items)} → 检查阈值或知识库覆盖")
    print("  └────────────────────────────────────────────────────────────┘")
    print()


def main():
    parser = argparse.ArgumentParser(description="QA Assistant 性能基准测试")
    parser.add_argument("--count", type=int, default=0,
                        help="随机抽取N条测试 (0=全部)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="QA匹配阈值 (默认0.85)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="禁用 cross-encoder reranking")
    parser.add_argument("--output", type=str, default="",
                        help="输出详细报告 JSON 路径")
    parser.add_argument("--test-file", type=str, default="",
                        help="测试集 JSON 文件路径")
    parser.add_argument("--label", type=str, default="",
                        help="报告标签 (如 'with-rerank', 'no-rerank')")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  QA Assistant 基准测试 v2")
    print(f"  阈值: {args.threshold}  |  Rerank: {'OFF' if args.no_rerank else 'ON'}")
    print("=" * 72)

    # Load test questions
    all_questions = load_questions(args.test_file)
    
    # Sample if --count specified
    if args.count > 0 and args.count < len(all_questions):
        questions = sample_questions(all_questions, args.count, args.seed)
        print(f"\n  从 {len(all_questions)} 条中随机抽取 {len(questions)} 条 (seed={args.seed})\n")
    else:
        questions = all_questions
        print(f"\n  加载全部 {len(questions)} 条测试问题\n")

    # Initialize searcher
    print("  初始化 HybridSearcher (加载模型中)...")
    t0 = time.perf_counter()
    searcher = HybridSearcher(
        match_threshold=args.threshold,
        enable_rag=True,
        use_rerank=not args.no_rerank,
    )
    init_ms = round((time.perf_counter() - t0) * 1000)
    print(f"  初始化完成 ({init_ms}ms)\n")

    # Run benchmark
    results = run_benchmark(questions, searcher)

    # Compute statistics
    stats = compute_statistics(results)
    stats["init_time_ms"] = init_ms
    stats["config"] = {
        "threshold": args.threshold,
        "rerank": not args.no_rerank,
        "count": args.count,
        "seed": args.seed,
        "total_questions": len(all_questions),
        "timestamp": datetime.now().isoformat(),
    }

    # Print report
    print_report(results, stats, label=args.label)

    # Save detailed report
    output_path = args.output or str(Path(__file__).parent.parent / "data" / "benchmark_report.json")
    report = {
        "config": stats["config"],
        "statistics": stats,
        "details": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  详细报告已保存: {output_path}")
    print()

    return stats


if __name__ == "__main__":
    main()
