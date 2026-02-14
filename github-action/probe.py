#!/usr/bin/env python3
"""
Oculus AI — Recommendability Probe
GitHub Action entrypoint that probes AI platforms and outputs scores.
"""

import os
import sys
import json
import time
import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin


# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE = os.environ.get("OCULUS_API_BASE", "https://api.oculus.ai/v2")
API_KEY = os.environ.get("INPUT_API_KEY", "")
TARGET_URL = os.environ.get("INPUT_URL", "")
PROMPTS = os.environ.get("INPUT_PROMPTS", "")
PLATFORMS = os.environ.get("INPUT_PLATFORMS", "chatgpt,claude,perplexity,gemini")
THRESHOLD = int(os.environ.get("INPUT_THRESHOLD", "0"))
FAIL_ON_DROP = float(os.environ.get("INPUT_FAIL_ON_DROP", "0"))
COMPETITORS = os.environ.get("INPUT_COMPARE_COMPETITORS", "")
OUTPUT_FORMAT = os.environ.get("INPUT_OUTPUT_FORMAT", "summary")
GITHUB_OUTPUT = os.environ.get("GITHUB_OUTPUT", "")
GITHUB_STEP_SUMMARY = os.environ.get("GITHUB_STEP_SUMMARY", "")


# ─── LLM Platform Probers ────────────────────────────────────────────────────

class PlatformProber:
    """Base class for AI platform probing."""

    PLATFORM_CONFIGS = {
        "chatgpt": {
            "name": "ChatGPT (OpenAI)",
            "api_url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-4o-mini",
            "env_key": "OPENAI_API_KEY",
        },
        "claude": {
            "name": "Claude (Anthropic)",
            "api_url": "https://api.anthropic.com/v1/messages",
            "model": "claude-3-5-haiku-20241022",
            "env_key": "ANTHROPIC_API_KEY",
        },
        "perplexity": {
            "name": "Perplexity AI",
            "api_url": "https://api.perplexity.ai/chat/completions",
            "model": "sonar",
            "env_key": "PERPLEXITY_API_KEY",
        },
        "gemini": {
            "name": "Google Gemini",
            "api_url": "https://generativelanguage.googleapis.com/v1beta/models",
            "model": "gemini-2.0-flash",
            "env_key": "GOOGLE_API_KEY",
        },
        "copilot": {
            "name": "Microsoft Copilot",
            "api_url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-4o-mini",
            "env_key": "AZURE_API_KEY",
        },
        "mistral": {
            "name": "Mistral AI",
            "api_url": "https://api.mistral.ai/v1/chat/completions",
            "model": "mistral-small-latest",
            "env_key": "MISTRAL_API_KEY",
        },
        "deepseek": {
            "name": "DeepSeek",
            "api_url": "https://api.deepseek.com/v1/chat/completions",
            "model": "deepseek-chat",
            "env_key": "DEEPSEEK_API_KEY",
        },
    }

    @staticmethod
    def _api_call(url: str, headers: dict, payload: dict, timeout: int = 30) -> dict:
        """Make an API call and return parsed JSON."""
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, Exception) as e:
            return {"error": str(e)}

    @classmethod
    def probe_openai_compatible(
        cls, platform_key: str, prompt: str, brand_url: str
    ) -> dict:
        """Probe OpenAI-compatible APIs (ChatGPT, Copilot, Mistral, DeepSeek, Perplexity)."""
        config = cls.PLATFORM_CONFIGS[platform_key]
        api_key = os.environ.get(config["env_key"], "")
        if not api_key:
            return {
                "platform": platform_key,
                "status": "skipped",
                "reason": f"No {config['env_key']} set",
            }

        system_msg = (
            "You are an AI assistant helping users find the best products and services. "
            "Answer naturally and recommend specific companies when relevant. "
            "Include URLs or company names you trust."
        )
        user_msg = f"{prompt}\n\n(Context: I'm researching options related to {brand_url})"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # Perplexity and Mistral use same OpenAI-compatible format
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
        }

        result = cls._api_call(config["api_url"], headers, payload)

        if "error" in result:
            return {
                "platform": platform_key,
                "status": "error",
                "error": result["error"],
            }

        response_text = ""
        if "choices" in result and result["choices"]:
            response_text = result["choices"][0].get("message", {}).get("content", "")

        return {
            "platform": platform_key,
            "platform_name": config["name"],
            "status": "success",
            "response": response_text,
            "model": config["model"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def probe_claude(cls, prompt: str, brand_url: str) -> dict:
        """Probe Anthropic Claude API."""
        config = cls.PLATFORM_CONFIGS["claude"]
        api_key = os.environ.get(config["env_key"], "")
        if not api_key:
            return {
                "platform": "claude",
                "status": "skipped",
                "reason": f"No {config['env_key']} set",
            }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": config["model"],
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n\n(Context: researching options related to {brand_url})",
                }
            ],
        }

        result = cls._api_call(config["api_url"], headers, payload)

        if "error" in result:
            return {"platform": "claude", "status": "error", "error": result["error"]}

        response_text = ""
        if "content" in result and result["content"]:
            response_text = result["content"][0].get("text", "")

        return {
            "platform": "claude",
            "platform_name": config["name"],
            "status": "success",
            "response": response_text,
            "model": config["model"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def probe_gemini(cls, prompt: str, brand_url: str) -> dict:
        """Probe Google Gemini API."""
        config = cls.PLATFORM_CONFIGS["gemini"]
        api_key = os.environ.get(config["env_key"], "")
        if not api_key:
            return {
                "platform": "gemini",
                "status": "skipped",
                "reason": f"No {config['env_key']} set",
            }

        url = f"{config['api_url']}/{config['model']}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{prompt}\n\n(Context: researching options related to {brand_url})"
                        }
                    ]
                }
            ],
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7},
        }

        result = cls._api_call(url, headers, payload)

        if "error" in result:
            return {"platform": "gemini", "status": "error", "error": result["error"]}

        response_text = ""
        if "candidates" in result and result["candidates"]:
            parts = result["candidates"][0].get("content", {}).get("parts", [])
            if parts:
                response_text = parts[0].get("text", "")

        return {
            "platform": "gemini",
            "platform_name": config["name"],
            "status": "success",
            "response": response_text,
            "model": config["model"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def probe(cls, platform_key: str, prompt: str, brand_url: str) -> dict:
        """Route to correct prober."""
        if platform_key == "claude":
            return cls.probe_claude(prompt, brand_url)
        elif platform_key == "gemini":
            return cls.probe_gemini(prompt, brand_url)
        elif platform_key in cls.PLATFORM_CONFIGS:
            return cls.probe_openai_compatible(platform_key, prompt, brand_url)
        else:
            return {"platform": platform_key, "status": "unknown", "error": "Unknown platform"}


# ─── Scoring Engine ──────────────────────────────────────────────────────────

class ScoringEngine:
    """Analyze probe responses and compute recommendability scores."""

    @staticmethod
    def extract_brand_from_url(url: str) -> str:
        """Extract brand name from URL."""
        import re
        # Remove protocol and www
        clean = re.sub(r"https?://(www\.)?", "", url)
        # Get domain without TLD
        domain = clean.split("/")[0].split(".")[0]
        return domain.lower()

    @staticmethod
    def analyze_response(response: str, brand: str, url: str) -> dict:
        """Analyze a single LLM response for brand mentions and sentiment."""
        response_lower = response.lower()
        brand_lower = brand.lower()

        # Count mentions
        mention_count = response_lower.count(brand_lower)

        # Check for URL mentions
        url_mentioned = url.lower().replace("https://", "").replace("http://", "").replace("www.", "") in response_lower

        # Position analysis (where in response is brand first mentioned)
        first_pos = response_lower.find(brand_lower)
        position_score = 0
        if first_pos >= 0:
            # Earlier mention = higher score
            position_score = max(0, 100 - (first_pos / max(len(response), 1)) * 100)

        # Recommendation language detection
        rec_phrases = [
            "i recommend", "i suggest", "consider using", "you should try",
            "great option", "top choice", "best option", "highly recommend",
            "excellent choice", "worth checking out", "stands out",
            "leader in", "known for", "trusted by", "popular choice"
        ]
        rec_score = sum(1 for phrase in rec_phrases if phrase in response_lower)

        # Negative sentiment detection
        neg_phrases = [
            "not recommended", "avoid", "drawback", "limitation",
            "competitor to", "alternative to", "unlike", "falls short"
        ]
        neg_count = sum(1 for phrase in neg_phrases if phrase in response_lower and brand_lower in response_lower)

        # Citation extraction
        url_pattern = r'https?://[^\s\)\]\"\'<>]+'
        citations = re.findall(url_pattern, response)

        # Compute component scores
        mention_score = min(100, mention_count * 25)  # 4+ mentions = 100
        recommendation_score = min(100, rec_score * 20)  # 5+ phrases = 100
        negativity_penalty = min(50, neg_count * 15)

        # Composite
        composite = (
            mention_score * 0.3
            + position_score * 0.25
            + recommendation_score * 0.3
            + (50 if url_mentioned else 0) * 0.15
            - negativity_penalty
        )

        return {
            "mention_count": mention_count,
            "url_mentioned": url_mentioned,
            "position_score": round(position_score, 1),
            "mention_score": round(mention_score, 1),
            "recommendation_score": round(recommendation_score, 1),
            "negativity_penalty": round(negativity_penalty, 1),
            "composite": round(max(0, min(100, composite)), 1),
            "citations": citations,
            "recommended": mention_count > 0 and rec_score > 0,
        }

    @staticmethod
    def generate_actions(scores: List[dict], brand: str) -> List[dict]:
        """Generate top 3 recommended actions based on probe results."""
        actions = []

        # Check mention gaps
        platforms_without_mentions = [
            s["platform"] for s in scores
            if s.get("analysis", {}).get("mention_count", 0) == 0
            and s.get("status") == "success"
        ]
        if platforms_without_mentions:
            actions.append({
                "action": f"Create structured data (FAQ schema, HowTo, Organization) to improve entity recognition",
                "impact": "high",
                "predicted_uplift": "+15-25%",
                "platforms_affected": platforms_without_mentions,
            })

        # Check citation gaps
        platforms_without_citations = [
            s["platform"] for s in scores
            if not s.get("analysis", {}).get("url_mentioned", False)
            and s.get("status") == "success"
        ]
        if platforms_without_citations:
            actions.append({
                "action": f"Publish authoritative content with clear expertise signals (author bios, citations, data)",
                "impact": "high",
                "predicted_uplift": "+10-20%",
                "platforms_affected": platforms_without_citations,
            })

        # Check recommendation language
        low_rec_platforms = [
            s["platform"] for s in scores
            if s.get("analysis", {}).get("recommendation_score", 0) < 30
            and s.get("status") == "success"
        ]
        if low_rec_platforms:
            actions.append({
                "action": f"Build comparison content and 'best of' guides that position {brand} as a solution",
                "impact": "medium",
                "predicted_uplift": "+8-15%",
                "platforms_affected": low_rec_platforms,
            })

        # Always suggest monitoring
        if len(actions) < 3:
            actions.append({
                "action": "Set up weekly AI probe monitoring to track score trends and react to drops",
                "impact": "medium",
                "predicted_uplift": "preventive",
                "platforms_affected": ["all"],
            })

        return actions[:3]


# ─── Output Formatters ────────────────────────────────────────────────────────

def set_output(name: str, value: str):
    """Set GitHub Action output variable."""
    if GITHUB_OUTPUT:
        with open(GITHUB_OUTPUT, "a") as f:
            # Handle multiline values
            if "\n" in str(value):
                delimiter = f"EOF_{hashlib.md5(name.encode()).hexdigest()[:8]}"
                f.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")
            else:
                f.write(f"{name}={value}\n")


def write_summary(content: str):
    """Write to GitHub Actions step summary."""
    if GITHUB_STEP_SUMMARY:
        with open(GITHUB_STEP_SUMMARY, "a") as f:
            f.write(content)


def format_summary_markdown(
    brand: str,
    url: str,
    platform_results: List[dict],
    composite_score: float,
    actions: List[dict],
    status: str,
) -> str:
    """Generate markdown summary for GitHub Actions."""
    icon = "✅" if status == "pass" else "❌"
    lines = [
        f"## {icon} Oculus AI Recommendability Report",
        f"",
        f"**Brand:** {brand} | **URL:** {url} | **Score:** {composite_score}/100",
        f"",
        f"### Platform Results",
        f"",
        f"| Platform | Status | Mentioned | Score | Recommended |",
        f"|----------|--------|-----------|-------|-------------|",
    ]

    for r in platform_results:
        if r["status"] == "success":
            a = r.get("analysis", {})
            mentioned = "✅" if a.get("mention_count", 0) > 0 else "❌"
            rec = "✅" if a.get("recommended", False) else "➖"
            lines.append(
                f"| {r.get('platform_name', r['platform'])} | ✅ | {mentioned} | {a.get('composite', 0)} | {rec} |"
            )
        elif r["status"] == "skipped":
            lines.append(
                f"| {r.get('platform_name', r['platform'])} | ⏭️ Skipped | — | — | — |"
            )
        else:
            lines.append(
                f"| {r.get('platform_name', r['platform'])} | ❌ Error | — | — | — |"
            )

    lines.extend([
        f"",
        f"### Top Actions",
        f"",
    ])
    for i, action in enumerate(actions, 1):
        lines.append(
            f"{i}. **{action['action']}** (Impact: {action['impact']}, Uplift: {action['predicted_uplift']})"
        )

    lines.extend([
        f"",
        f"---",
        f"*Powered by [Oculus AI](https://oculus-ai.dev) — The Stripe of AI Visibility*",
    ])

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("🔮 Oculus AI — Recommendability Check")
    print("=" * 50)

    # Validate inputs
    if not TARGET_URL:
        print("::error::Missing required input: url")
        sys.exit(1)

    # Extract brand
    brand = ScoringEngine.extract_brand_from_url(TARGET_URL)
    print(f"📌 Brand: {brand}")
    print(f"🌐 URL: {TARGET_URL}")

    # Parse platforms
    platforms = [p.strip().lower() for p in PLATFORMS.split(",") if p.strip()]
    print(f"🤖 Platforms: {', '.join(platforms)}")

    # Parse or generate prompts
    if PROMPTS:
        prompts = [p.strip() for p in PROMPTS.split(",") if p.strip()]
    else:
        prompts = [
            f"What are the best {brand}-related tools available?",
            f"Can you recommend a good {brand} alternative or competitor?",
            f"What companies should I consider for services like {brand}?",
        ]
    print(f"💬 Prompts: {len(prompts)}")

    # Run probes
    all_results = []
    for platform in platforms:
        print(f"\n🔍 Probing {platform}...")
        for prompt in prompts:
            result = PlatformProber.probe(platform, prompt, TARGET_URL)

            if result["status"] == "success":
                # Analyze the response
                analysis = ScoringEngine.analyze_response(
                    result["response"], brand, TARGET_URL
                )
                result["analysis"] = analysis
                result["prompt"] = prompt
                print(f"   ✅ Score: {analysis['composite']} | Mentions: {analysis['mention_count']}")
            elif result["status"] == "skipped":
                print(f"   ⏭️  Skipped: {result.get('reason', 'No API key')}")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown')}")

            all_results.append(result)
            time.sleep(0.5)  # Rate limiting

    # Aggregate scores
    successful = [r for r in all_results if r["status"] == "success"]
    if successful:
        scores = [r["analysis"]["composite"] for r in successful]
        composite_score = round(sum(scores) / len(scores), 1)
        mention_count = sum(
            1 for r in successful if r["analysis"]["mention_count"] > 0
        )
    else:
        composite_score = 0
        mention_count = 0

    total_platforms = len([r for r in all_results if r["status"] != "unknown"])

    # Generate actions
    actions = ScoringEngine.generate_actions(successful, brand)

    # Determine pass/fail
    status = "pass"
    fail_reason = ""

    if THRESHOLD > 0 and composite_score < THRESHOLD:
        status = "fail"
        fail_reason = f"Score {composite_score} below threshold {THRESHOLD}"

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"📊 Composite Score: {composite_score}/100")
    print(f"📣 Mentioned by: {mention_count}/{total_platforms} platforms")
    print(f"{'✅ PASS' if status == 'pass' else '❌ FAIL'}{': ' + fail_reason if fail_reason else ''}")

    # Set outputs
    set_output("score", str(composite_score))
    set_output("previous_score", "n/a")
    set_output("score_delta", "0")
    set_output("mention_count", str(mention_count))
    set_output("total_platforms", str(total_platforms))
    set_output("entity_centrality", "n/a")
    set_output("top_actions", json.dumps(actions))
    set_output("status", status)

    # Write step summary
    summary_md = format_summary_markdown(
        brand, TARGET_URL, all_results, composite_score, actions, status
    )
    write_summary(summary_md)

    # Write JSON report
    report = {
        "brand": brand,
        "url": TARGET_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "composite_score": composite_score,
        "mention_count": mention_count,
        "total_platforms": total_platforms,
        "status": status,
        "platform_results": all_results,
        "actions": actions,
    }

    report_path = "/tmp/oculus-report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Full report: {report_path}")

    if OUTPUT_FORMAT == "json":
        print(json.dumps(report, indent=2, default=str))

    # Exit with failure if needed
    if status == "fail":
        print(f"\n::error::{fail_reason}")
        sys.exit(1)

    print("\n🎯 Done! AI recommendability check complete.")


if __name__ == "__main__":
    main()
