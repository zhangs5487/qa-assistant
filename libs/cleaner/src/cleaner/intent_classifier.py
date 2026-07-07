"""Intent classifier for query routing.

Identifies the user's intent BEFORE dispatching to QA/RAG.
Used to adjust matching threshold per intent type.

Intents:
    content_query    - What's in X? What does X provide?
    policy_faq       - Policy questions, subsidies, conditions
    how_to           - How to register, apply, use something
    document_lookup  - Specific document/PDF/report lookup
    platform_info    - What is this platform? What can it do?
    general          - Other / unclear
"""

import logging
from typing import Literal

import numpy as np

from config.settings import settings
from llm.factory import create_embedding_provider

logger = logging.getLogger(__name__)

IntentType = Literal[
    "content_query",
    "policy_faq",
    "how_to",
    "document_lookup",
    "platform_info",
    "general",
]

# Keyword signals per intent (fast path)
KEYWORD_MAP: dict[IntentType, list[str]] = {
    "content_query": [
        "有什么", "提供什么", "包含什么", "内容", "包括",
        "有哪些", "都有什么", "可以做什么",
    ],
    "how_to": [
        "怎么", "如何", "怎样", "步骤", "流程", "方法",
        "申请", "注册", "登录", "开通", "使用",
        "操作", "办理", "报名",
    ],
    "document_lookup": [
        "文件", "文档", "PDF", "pdf", "附件", "下载",
        "原文", "全文", "通知", "公告",
    ],
    "policy_faq": [
        "政策", "补贴", "扶持", "奖励", "优惠", "资助",
        "条件", "要求", "标准", "资格", "入驻",
        "入驻", "申请条件", "多少钱",
    ],
    "platform_info": [
        "是什么", "介绍", "关于", "功能",
        "服务", "用途", "这是一个",
    ],
}


class IntentClassifier:
    """Classify user query intent for proper routing.

    Uses keyword heuristics first (fast path), then falls back to
    embedding-based similarity against exemplar questions.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder
        self._exemplar_embeddings: dict[IntentType, list[np.ndarray]] = {}
        self._init_exemplars()

    # ---- Exemplar questions per intent ----

    EXEMPLARS: dict[IntentType, list[str]] = {
        "content_query": [
            "行业动态里有什么内容",
            "政策中心发布了什么",
            "智能体工坊提供什么服务",
            "平台有哪些模型可以用",
            "数据集市场有什么数据",
            "模型广场有哪些模型",
            "最近有什么活动",
            "平台提供哪些GPU型号",
            "平台模型广场有哪些大语言模型",
            "供需市场上有哪些AI解决方案",
            "有什么比赛可以参加",
            "有没有中文版权图书数据",
            "数龙杯有哪些赛道",
            "满天星挑战赛的主题是什么",
            "竞赛有什么参赛要求",
        ],
        "how_to": [
            "怎么注册平台账号",
            "如何申请算力资源",
            "怎样申请政策补贴",
            "入驻流程是什么",
            "模型怎么调用",
            "如何下载数据集",
            "怎么充值算力账户",
            "如何申请API Key",
            "忘记密码了怎么办",
            "怎么在平台发布需求",
            "怎么报名参加比赛",
            "数龙杯大学生怎么报名",
            "竞赛报名渠道有哪些",
            "如何参加满天星挑战赛",
        ],
        "policy_faq": [
            "申请入驻需要什么条件",
            "政策补贴多久到账",
            "算力补贴可以叠加吗",
            "哪些企业可以享受扶持",
            "入驻后有什么权益",
            "满天星行动计划有什么奖励",
            "平台的政策扶持有哪些",
            "算力券补贴比例多少",
            "新型智能产品最高奖励多少",
            "申请政策需要什么材料",
        ],
        "document_lookup": [
            "满天星行动计划文件",
            "查看政策原文",
            "补贴办法PDF下载",
            "通知文件在哪里看",
            "政策全文链接",
            "算力基础设施行动计划原文",
            "找一下关于AI赋能的通知文件",
        ],
        "platform_info": [
            "这个平台是什么",
            "平台提供哪些功能",
            "重庆市AI平台能做什么",
            "平台有哪些服务",
            "平台是干什么的",
            "介绍一下这个平台",
            "平台背景是什么",
        ],
    }

    def _init_exemplars(self):
        """Embed exemplar questions for later similarity comparison."""
        if self.embedder is None:
            try:
                self.embedder = create_embedding_provider()
            except Exception as e:
                logger.warning("Embedder unavailable for intent classification: %s", e)
                return

        for intent, questions in self.EXEMPLARS.items():
            try:
                embs = self.embedder.embed_documents(questions)
                self._exemplar_embeddings[intent] = [
                    np.array(e, dtype=np.float32) for e in embs
                ]
            except Exception as e:
                logger.warning("Failed to embed exemplars for %s: %s", intent, e)

    # ---- Classification ----

    def classify(self, query: str) -> tuple[IntentType, float]:
        """Classify the intent of a user query.

        Returns:
            (intent, confidence) where confidence is 0.0–1.0.
        """
        query_lower = query.lower().strip()

        # 1. Fast path: keyword matching
        keyword_scores: dict[IntentType, int] = {}
        for intent, keywords in KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in query)
            if score > 0:
                keyword_scores[intent] = score

        if keyword_scores:
            best_keyword_intent = max(keyword_scores, key=keyword_scores.get)
            max_score = keyword_scores[best_keyword_intent]
            # High confidence if 2+ keywords match
            if max_score >= 2:
                return best_keyword_intent, 0.85
            # Medium confidence if 1 keyword matches AND no other intents
            if max_score == 1 and len(keyword_scores) == 1:
                return best_keyword_intent, 0.7

        # 2. Slow path: embedding similarity against exemplars
        if self.embedder and self._exemplar_embeddings:
            try:
                query_emb = np.array(self.embedder.embed_query(query), dtype=np.float32)
                q_norm = np.linalg.norm(query_emb)
                if q_norm > 0:
                    query_emb = query_emb / q_norm

                    best_intent: IntentType = "general"
                    best_sim = 0.0

                    for intent, emb_list in self._exemplar_embeddings.items():
                        for exemplar_emb in emb_list:
                            e_norm = np.linalg.norm(exemplar_emb)
                            if e_norm > 0:
                                # query_emb already normalized above; only divide by e_norm
                                sim = float(np.dot(query_emb, exemplar_emb) / e_norm)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_intent = intent

                    if best_sim > 0.75:
                        return best_intent, best_sim
                    if best_sim > 0.6:
                        return best_intent, 0.6
            except Exception as e:
                logger.debug("Embedding-based classification failed: %s", e)

        # 3. Fallback
        return "general", 0.3

    def classify_batch(self, queries: list[str]) -> list[tuple[IntentType, float]]:
        """Classify multiple queries at once."""
        return [self.classify(q) for q in queries]
