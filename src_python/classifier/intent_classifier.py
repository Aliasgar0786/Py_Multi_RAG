"""
IntentClassifier - Analyzes user queries and determines intent + target RAG.

Phase 1 implementation: Rule-based keyword matching.

Architecture is designed so the classification strategy can be swapped
(e.g., to embedding-based, ML, or LLM classification) by implementing
the ClassificationStrategy interface — no changes to the rest of the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

# Supported platform contexts
Platform = Literal["consumer", "merchant", "admin"]


@dataclass
class ClassificationResult:
    """The result of classifying a user query."""
    # The platform the query came from
    platform: Platform
    # The detected user intent (e.g., "refunds", "product_management")
    intent: str
    # Which RAG module should handle this query
    target_rag: str  # "customer_rag" | "merchant_rag" | "admin_rag"
    # Confidence score of the classification (0.0 - 1.0)
    confidence: float


class ClassificationStrategy(ABC):
    """
    Strategy interface — swap this to change classification approach
    without modifying the IntentClassifier class.
    """

    @abstractmethod
    def classify(self, platform: Platform, query: str) -> ClassificationResult:
        ...


# ─────────────────────────────────────────────────────────────────
# Rule-Based Classification Strategy (Phase 1)
# ─────────────────────────────────────────────────────────────────

@dataclass
class _KeywordRule:
    intent: str
    keywords: list[str]


_CONSUMER_RULES: list[_KeywordRule] = [
    _KeywordRule(
        intent="refunds",
        keywords=["refund", "return", "money back", "reimburse", "refund policy", "return policy"],
    ),
    _KeywordRule(
        intent="orders",
        keywords=["order", "track", "shipping", "delivery", "cancel order", "order status", "shipped", "tracking"],
    ),
    _KeywordRule(
        intent="shopping",
        keywords=["buy", "purchase", "cart", "checkout", "price", "deal", "discount", "coupon", "product", "search", "browse"],
    ),
    _KeywordRule(
        intent="account",
        keywords=["account", "profile", "password", "login", "sign up", "register", "settings", "privacy"],
    ),
    _KeywordRule(
        intent="support",
        keywords=["help", "support", "contact", "complaint", "issue", "problem"],
    ),
]

_MERCHANT_RULES: list[_KeywordRule] = [
    _KeywordRule(
        intent="onboarding",
        keywords=["onboard", "register", "apply", "application", "become seller", "start selling", "seller account", "sign up as seller", "verification"],
    ),
    _KeywordRule(
        intent="product_management",
        keywords=["product", "listing", "inventory", "stock", "sku", "upload", "catalog", "add product", "create listing", "variant", "category"],
    ),
    _KeywordRule(
        intent="store_management",
        keywords=["store", "dashboard", "analytics", "sales", "report", "performance", "metrics", "shipping", "fulfillment", "payout", "commission"],
    ),
    _KeywordRule(
        intent="customer_handling",
        keywords=["customer", "message", "review", "return", "refund", "dispute", "response time"],
    ),
]

_ADMIN_RULES: list[_KeywordRule] = [
    _KeywordRule(
        intent="user_management",
        keywords=["user", "account", "suspend", "ban", "reset password", "manage user", "activity log"],
    ),
    _KeywordRule(
        intent="content_moderation",
        keywords=["moderate", "flag", "review", "violation", "content", "counterfeit", "spam", "guidelines"],
    ),
    _KeywordRule(
        intent="monitoring",
        keywords=["monitor", "alert", "health", "uptime", "traffic", "incident", "outage", "performance", "error rate"],
    ),
    _KeywordRule(
        intent="workflows",
        keywords=["workflow", "process", "procedure", "escalat", "approval", "campaign", "audit", "compliance", "report"],
    ),
    _KeywordRule(
        intent="disputes",
        keywords=["dispute", "resolution", "escalat", "investigate"],
    ),
]


class RuleBasedStrategy(ClassificationStrategy):
    def classify(self, platform: Platform, query: str) -> ClassificationResult:
        normalized_query = query.lower()

        # Select rules based on platform — this ensures context isolation
        if platform == "consumer":
            rules = _CONSUMER_RULES
            target_rag = "customer_rag"
        elif platform == "merchant":
            rules = _MERCHANT_RULES
            target_rag = "merchant_rag"
        elif platform == "admin":
            rules = _ADMIN_RULES
            target_rag = "admin_rag"
        else:
            return ClassificationResult(
                platform=platform,
                intent="general",
                target_rag="customer_rag",
                confidence=0.1,
            )

        # Score each rule by counting keyword matches
        best_intent = "general"
        best_score = 0

        for rule in rules:
            score = 0
            for keyword in rule.keywords:
                if keyword in normalized_query:
                    # Multi-word keywords get higher score
                    score += len(keyword.split(" "))
            if score > best_score:
                best_score = score
                best_intent = rule.intent

        # Normalize confidence
        confidence = min(best_score / 4, 1.0) if best_score > 0 else 0.3

        return ClassificationResult(
            platform=platform,
            intent=best_intent,
            target_rag=target_rag,
            confidence=round(confidence * 100) / 100,
        )


# ─────────────────────────────────────────────────────────────────
# IntentClassifier (strategy-agnostic wrapper)
# ─────────────────────────────────────────────────────────────────

class IntentClassifier:
    def __init__(self, strategy: ClassificationStrategy | None = None):
        # Default to rule-based for Phase 1; swap strategy for future upgrades
        self._strategy = strategy or RuleBasedStrategy()
        print(
            f"[IntentClassifier] Initialized with strategy: "
            f"{self._strategy.__class__.__name__}"
        )

    def set_strategy(self, strategy: ClassificationStrategy) -> None:
        """
        Replace the classification strategy at runtime.
        Allows upgrading from rule-based to ML/LLM without restarting.
        """
        self._strategy = strategy
        print(
            f"[IntentClassifier] Strategy updated to: "
            f"{strategy.__class__.__name__}"
        )

    def classify(self, platform: Platform, query: str) -> ClassificationResult:
        """
        Classify a user query based on platform context and query content.
        """
        result = self._strategy.classify(platform, query)
        print(
            f'[IntentClassifier] Platform: {platform} | Query: "{query}" '
            f"-> Intent: {result.intent} | RAG: {result.target_rag} | "
            f"Confidence: {result.confidence}"
        )
        return result
