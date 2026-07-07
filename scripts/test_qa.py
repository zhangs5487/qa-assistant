"""QA + RAG 测试脚本 — 含意图分析、来源链接

Usage:
    python scripts/test_qa.py                    # 交互模式
    python scripts/test_qa.py -q "..."            # 单条
    python scripts/test_qa.py -b questions.txt    # 批量
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "shared", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "llm", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "cleaner", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "storage", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

from cleaner.hybrid_retriever import HybridRetriever
from cleaner.intent_classifier import IntentClassifier
from config.settings import settings
from llm.factory import create_chat_provider


RAG_PROMPT = """你是重庆市人工智能公共服务平台的AI助手。请根据参考资料回答用户问题。

要求：
- 基于参考资料回答，把资料中已有的相关信息整理出来
- 如果资料中有部分信息但不够完整，先给出已有的信息，再说明哪些部分未提及
- 不要编造参考资料中没有的信息
- 注明信息来源（标题/文档名称）

参考资料：
{context}

用户问题：{question}

回答："""


class HybridSearcher:
    def __init__(self, match_threshold: float = 0.85, enable_rag: bool = True,
                 use_rerank: bool = True):
        self.match_threshold = match_threshold
        self.enable_rag = enable_rag
        self.use_rerank = use_rerank
        self.chatter = create_chat_provider() if enable_rag else None
        # 共享同一个 embedder，避免 BGE-M3 加载两遍
        from llm.factory import create_embedding_provider
        _shared_embedder = create_embedding_provider()
        self.intent_clf = IntentClassifier(embedder=_shared_embedder)
        self.retriever = HybridRetriever(use_rerank=use_rerank, embedder=_shared_embedder)

    # ---- RAG ----

    def rag_generate(self, question: str, chunks: list[dict]) -> str:
        if not self.chatter or not chunks:
            return "(RAG不可用)"
        parts = []
        for i, c in enumerate(chunks[:5]):
            title = c.get("title", "")
            text = c.get("content", "")
            if title:
                parts.append("[%d] %s\n%s" % (i + 1, title, text[:500]))
            else:
                parts.append("[%d] %s" % (i + 1, text[:500]))
        context = "\n\n".join(parts)
        prompt = RAG_PROMPT.format(context=context, question=question)
        from llm.base import ChatConfig, ChatMessage
        try:
            return self.chatter.chat(
                messages=[ChatMessage(role="user", content=prompt)],
                config=ChatConfig(temperature=0.3, max_tokens=2048),
            )
        except Exception as e:
            return "(RAG生成失败: %s)" % e

    # ---- Intent-Aware Routing ----

    def ask(self, question: str) -> dict:
        t0 = time.time()

        # 并行：意图分析 + QA检索（共享 BGE-M3，互不阻塞）
        with ThreadPoolExecutor(max_workers=2) as pool:
            intent_fut = pool.submit(self.intent_clf.classify, question)
            qa_fut = pool.submit(self.retriever.search_qa, question)
            intent, intent_conf = intent_fut.result()
            qa_results = qa_fut.result()
        best_qa = qa_results[0] if qa_results else None

        # Find max similarity across all results (reranker may reorder BM25-only to top)
        max_sim = max((r.get("similarity", 0) for r in qa_results), default=0)
        max_rerank = max((r.get("rerank_score", 0) for r in qa_results), default=0)

        result = {
            "question": question,
            "intent": intent,
            "intent_confidence": intent_conf,
            "mode": "QA",
            "qa_hit": False,
            "qa_similarity": max(max_sim, max_rerank),
            "qa_question": best_qa["question"] if best_qa else "",
            "qa_answer": best_qa["answer"] if best_qa else "",
            "qa_source": best_qa.get("source", "") if best_qa else "",
            "rag_answer": "",
            "rag_chunks": [],
            "latency_ms": 0,
        }

        # Step 3: Decide if QA hit (intent-aware threshold)
        threshold_map = {
            "policy_faq": 0.85,
            "how_to": 0.83,
            "content_query": 0.80,
            "document_lookup": 0.82,
            "platform_info": 0.85,
            "general": self.match_threshold,
        }
        threshold = threshold_map.get(intent, self.match_threshold)

        if best_qa and result["qa_similarity"] >= threshold:
            result["qa_hit"] = True

        # Step 4: RAG fallback
        if not result["qa_hit"] and self.enable_rag:
            result["mode"] = "RAG"
            chunks = self.retriever.search_chunks(question, top_k=5, use_rerank=False)
            result["rag_chunks"] = chunks
            if chunks:
                result["rag_answer"] = self.rag_generate(question, chunks)

        result["latency_ms"] = int((time.time() - t0) * 1000)
        return result

    # ---- Source link helpers ----

    @staticmethod
    def make_policy_url(policy_id: str) -> str:
        """Build policy detail page URL from policy ID."""
        return "https://cqaip.cn/policies/" + policy_id

    def format_answer(self, result: dict) -> str:
        """Format the answer with intent, source links, and full info."""
        lines = []
        lines.append("")
        lines.append("  " + "-" * 56)

        # Intent display
        intent_label = {
            "content_query": "内容查询", "policy_faq": "政策咨询",
            "how_to": "操作指导", "document_lookup": "文件查阅", "platform_info": "平台介绍", "general": "通用"
        }.get(result["intent"], result["intent"])
        lines.append("  意图: [%s]  (%.2f)" % (intent_label, result["intent_confidence"]))

        if result["qa_hit"]:
            mode_tag = "QA HIT"
            if self.use_rerank:
                mode_tag += " + Re-rank"
            lines.append("  模式: %s  (相似度=%.4f)" % (mode_tag, result["qa_similarity"]))
            # Show the matched question
            if result["qa_question"] != result["question"]:
                lines.append("  匹配: %s" % result["qa_question"])
            lines.append("  " + "-" * 56)
            lines.append("  %s" % result["qa_answer"])
            # Source link
            if result.get("qa_source"):
                lines.append("")
                lines.append("  [来源] %s" % result["qa_source"])
        else:
            lines.append("  模式: RAG (QA最佳=%.4f)" % (result["qa_similarity"]))
            lines.append("  " + "-" * 56)

            if result["rag_chunks"]:
                lines.append("  [检索到%d个相关片段]" % len(result["rag_chunks"]))
                for i, c in enumerate(result["rag_chunks"][:3]):
                    src = c.get("source", "")
                    sim = c.get("similarity", 0)
                    if src:
                        lines.append("    [%d] %.4f  %s" % (i + 1, sim, src))
                    else:
                        lines.append("    [%d] %.4f" % (i + 1, sim))

                lines.append("")
                lines.append("  --- LLM回答 ---")
                lines.append("  %s" % result["rag_answer"])
                # 从 RAG chunks 提取有效来源链接
                seen_urls = set()
                source_urls = []
                for c in result["rag_chunks"]:
                    src = (c.get("source") or "").strip()
                    if not src or src in seen_urls:
                        continue
                    # 过滤无意义的内部路径（保留 /api/ 作为信息来源提示）
                    if src in ("", "/", "#"):
                        continue
                    seen_urls.add(src)
                    source_urls.append(src)
                if source_urls:
                    lines.append("")
                    for u in source_urls[:3]:  # 最多展示3条
                        lines.append("  [查看原文] %s" % u)
            else:
                lines.append("  (未检索到相关内容)")

        lines.append("  " + "-" * 56)
        lines.append("  耗时: %dms" % result["latency_ms"])
        lines.append("")

        return "\n".join(lines)


# ---- Interactive ----

def interactive_mode(searcher):
    print()
    print("=" * 60)
    print("  QA Assistant — 意图分析 + QA/RAG 双模式")
    print("  命令: exit | threshold 0.80 | rag on/off | rerank on/off | stats")
    print("=" * 60)
    while True:
        try:
            query = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        if query.lower().startswith("threshold"):
            try:
                searcher.match_threshold = float(query.split()[1])
                print("  threshold -> %.2f" % searcher.match_threshold)
            except Exception:
                print("  usage: threshold 0.85")
            continue
        if query.lower().startswith("rag"):
            val = query.split()[-1]
            if val in ("on", "off"):
                searcher.enable_rag = val == "on"
                print("  RAG: %s" % ("ON" if searcher.enable_rag else "OFF"))
            continue
        if query.lower().startswith("rerank"):
            val = query.split()[-1]
            if val in ("on", "off"):
                searcher.use_rerank = val == "on"
                searcher.retriever._use_rerank = searcher.use_rerank
                print("  Re-rank: %s" % ("ON" if searcher.use_rerank else "OFF"))
            continue
        if query.lower() == "stats":
            st = searcher.retriever.vector_store.stats()
            for n, s in st.items():
                print("  %s: %s" % (n, s))
            continue

        result = searcher.ask(query)
        print(searcher.format_answer(result))


# ---- Batch ----

def batch_mode(searcher, file_path):
    with open(file_path, encoding="utf-8") as f:
        questions = [l.strip() for l in f if l.strip()]

    qa_hits = rag_ok = 0
    for q in questions:
        r = searcher.ask(q)
        if r["qa_hit"]:
            qa_hits += 1
            tag = "[QA]"
        elif r["rag_answer"]:
            rag_ok += 1
            tag = "[RAG]"
        else:
            tag = "[MIS]"
        intent_tag = r["intent"][:6].ljust(6)
        print("  %s %s sim=%.4f  %s" % (tag, intent_tag, r["qa_similarity"], q[:50]))

    total = len(questions)
    resolved = qa_hits + rag_ok
    print("\n  QA:%d  RAG:%d  MISS:%d  解决率:%.0f%%" % (qa_hits, rag_ok, total - resolved, resolved / total * 100))
    json.dump({"threshold": searcher.match_threshold, "total": total, "qa_hits": qa_hits,
               "rag_answered": rag_ok}, open("qa_test_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ---- Main ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", type=str, default="")
    p.add_argument("-b", "--batch", type=str, default="")
    p.add_argument("-t", "--threshold", type=float, default=0.85)
    p.add_argument("--qa-only", action="store_true")
    p.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    if args.rebuild:
        from cleaner.qa_processor import run_full_qa_pipeline
        run_full_qa_pipeline()
        subprocess.run([sys.executable, str(Path(__file__).parent / "build_knowledge_base.py")], check=True)

    searcher = HybridSearcher(
        match_threshold=args.threshold,
        enable_rag=not args.qa_only,
        use_rerank=not args.no_rerank,
    )

    if args.batch:
        batch_mode(searcher, args.batch)
    elif args.question:
        result = searcher.ask(args.question)
        print(searcher.format_answer(result))
    else:
        interactive_mode(searcher)


if __name__ == "__main__":
    main()
