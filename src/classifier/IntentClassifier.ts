/**
 * IntentClassifier - Analyzes user queries and determines intent + target RAG.
 *
 * Phase 1 implementation: Rule-based keyword matching.
 *
 * Architecture is designed so the classification strategy can be swapped
 * (e.g., to embedding-based, ML, or LLM classification) by implementing
 * the ClassificationStrategy interface — no changes to the rest of the system.
 */

/** Supported platform contexts */
export type Platform = "consumer" | "merchant" | "admin";

/** The result of classifying a user query */
export interface ClassificationResult {
  /** The platform the query came from */
  platform: Platform;
  /** The detected user intent (e.g., "refunds", "product_management") */
  intent: string;
  /** Which RAG module should handle this query */
  targetRAG: "customer_rag" | "merchant_rag" | "admin_rag";
  /** Confidence score of the classification (0.0 - 1.0) */
  confidence: number;
}

/**
 * Strategy interface — swap this to change classification approach
 * without modifying the IntentClassifier class.
 */
export interface ClassificationStrategy {
  classify(platform: Platform, query: string): ClassificationResult;
}

// ─────────────────────────────────────────────────────────────────
// Rule-Based Classification Strategy (Phase 1)
// ─────────────────────────────────────────────────────────────────

interface KeywordRule {
  intent: string;
  keywords: string[];
}

const CONSUMER_RULES: KeywordRule[] = [
  {
    intent: "refunds",
    keywords: ["refund", "return", "money back", "reimburse", "refund policy", "return policy"],
  },
  {
    intent: "orders",
    keywords: ["order", "track", "shipping", "delivery", "cancel order", "order status", "shipped", "tracking"],
  },
  {
    intent: "shopping",
    keywords: ["buy", "purchase", "cart", "checkout", "price", "deal", "discount", "coupon", "product", "search", "browse"],
  },
  {
    intent: "account",
    keywords: ["account", "profile", "password", "login", "sign up", "register", "settings", "privacy"],
  },
  {
    intent: "support",
    keywords: ["help", "support", "contact", "complaint", "issue", "problem"],
  },
];

const MERCHANT_RULES: KeywordRule[] = [
  {
    intent: "onboarding",
    keywords: ["onboard", "register", "apply", "application", "become seller", "start selling", "seller account", "sign up as seller", "verification"],
  },
  {
    intent: "product_management",
    keywords: ["product", "listing", "inventory", "stock", "sku", "upload", "catalog", "add product", "create listing", "variant", "category"],
  },
  {
    intent: "store_management",
    keywords: ["store", "dashboard", "analytics", "sales", "report", "performance", "metrics", "shipping", "fulfillment", "payout", "commission"],
  },
  {
    intent: "customer_handling",
    keywords: ["customer", "message", "review", "return", "refund", "dispute", "response time"],
  },
];

const ADMIN_RULES: KeywordRule[] = [
  {
    intent: "user_management",
    keywords: ["user", "account", "suspend", "ban", "reset password", "manage user", "activity log"],
  },
  {
    intent: "content_moderation",
    keywords: ["moderate", "flag", "review", "violation", "content", "counterfeit", "spam", "guidelines"],
  },
  {
    intent: "monitoring",
    keywords: ["monitor", "alert", "health", "uptime", "traffic", "incident", "outage", "performance", "error rate"],
  },
  {
    intent: "workflows",
    keywords: ["workflow", "process", "procedure", "escalat", "approval", "campaign", "audit", "compliance", "report"],
  },
  {
    intent: "disputes",
    keywords: ["dispute", "resolution", "escalat", "investigate"],
  },
];

export class RuleBasedStrategy implements ClassificationStrategy {
  classify(platform: Platform, query: string): ClassificationResult {
    const normalizedQuery = query.toLowerCase();

    // Select rules based on platform — this ensures context isolation
    let rules: KeywordRule[];
    let targetRAG: "customer_rag" | "merchant_rag" | "admin_rag";

    switch (platform) {
      case "consumer":
        rules = CONSUMER_RULES;
        targetRAG = "customer_rag";
        break;
      case "merchant":
        rules = MERCHANT_RULES;
        targetRAG = "merchant_rag";
        break;
      case "admin":
        rules = ADMIN_RULES;
        targetRAG = "admin_rag";
        break;
      default:
        return {
          platform,
          intent: "general",
          targetRAG: "customer_rag",
          confidence: 0.1,
        };
    }

    // Score each rule by counting keyword matches
    let bestIntent = "general";
    let bestScore = 0;

    for (const rule of rules) {
      let score = 0;
      for (const keyword of rule.keywords) {
        if (normalizedQuery.includes(keyword)) {
          score += keyword.split(" ").length; // Multi-word keywords get higher score
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestIntent = rule.intent;
      }
    }

    // Normalize confidence
    const confidence = bestScore > 0 ? Math.min(bestScore / 4, 1.0) : 0.3;

    return {
      platform,
      intent: bestIntent,
      targetRAG,
      confidence: Math.round(confidence * 100) / 100,
    };
  }
}

// ─────────────────────────────────────────────────────────────────
// IntentClassifier (strategy-agnostic wrapper)
// ─────────────────────────────────────────────────────────────────

export class IntentClassifier {
  private strategy: ClassificationStrategy;

  constructor(strategy?: ClassificationStrategy) {
    // Default to rule-based for Phase 1; swap strategy for future upgrades
    this.strategy = strategy || new RuleBasedStrategy();
    console.log(
      `[IntentClassifier] Initialized with strategy: ${this.strategy.constructor.name}`
    );
  }

  /**
   * Replace the classification strategy at runtime.
   * Allows upgrading from rule-based to ML/LLM without restarting.
   */
  setStrategy(strategy: ClassificationStrategy): void {
    this.strategy = strategy;
    console.log(
      `[IntentClassifier] Strategy updated to: ${strategy.constructor.name}`
    );
  }

  /**
   * Classify a user query based on platform context and query content.
   */
  classify(platform: Platform, query: string): ClassificationResult {
    const result = this.strategy.classify(platform, query);
    console.log(
      `[IntentClassifier] Platform: ${platform} | Query: "${query}" → Intent: ${result.intent} | RAG: ${result.targetRAG} | Confidence: ${result.confidence}`
    );
    return result;
  }
}
