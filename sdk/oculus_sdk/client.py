"""
Oculus AI SDK — Client
The main entry point for interacting with the Oculus AI API.

Usage:
    from oculus_sdk import OculusClient

    client = OculusClient(api_key="ocu_live_...")

    # Quick probe
    result = client.probe("https://yoursite.com")
    print(f"Score: {result.score}/100")
    print(f"Mentioned by: {result.mentioned_by}/{result.total_platforms} platforms")

    # With specific platforms and prompts
    result = client.probe(
        url="https://yoursite.com",
        platforms=["chatgpt", "claude", "perplexity", "gemini"],
        prompts=["best CRM for startups", "top project management tools"],
        threshold=50,
    )

    if result.status == "fail":
        print("AI visibility dropped!")
        for action in result.actions:
            print(f"  → {action.action} ({action.predicted_uplift})")

    # Simulate uplift
    sim = client.simulate(
        url="https://yoursite.com",
        action="Add FAQ schema markup",
    )
    print(f"Predicted uplift: +{sim.uplift_percent}% ({sim.confidence_lower}-{sim.confidence_upper})")
"""

import json
import hashlib
import time
from typing import List, Optional, Dict, Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone

from oculus_sdk.models import (
    ProbeResult,
    PlatformResult,
    Action,
    ScoreReport,
    SimulationResult,
)


class OculusError(Exception):
    """Base exception for Oculus SDK errors."""
    def __init__(self, message: str, status_code: int = 0, response: str = ""):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class OculusAuthError(OculusError):
    """Authentication error — invalid or missing API key."""
    pass


class OculusRateLimitError(OculusError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s")


class OculusClient:
    """
    Oculus AI API Client.

    The main interface for probing AI platforms, checking recommendability scores,
    running simulations, and managing your AI visibility.

    Args:
        api_key: Your Oculus AI API key (starts with 'ocu_').
        base_url: API base URL (default: https://api.oculus.ai/v2).
        timeout: Request timeout in seconds (default: 30).
        max_retries: Maximum retry attempts on transient failures (default: 3).
    """

    DEFAULT_BASE_URL = "https://api.oculus.ai/v2"
    DEFAULT_PLATFORMS = ["chatgpt", "claude", "perplexity", "gemini"]

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        if not api_key:
            raise OculusAuthError("API key is required. Get yours at https://oculus-ai.dev")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    # ─── Core API Methods ────────────────────────────────────────────────

    def probe(
        self,
        url: str,
        platforms: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        threshold: int = 0,
        compare_competitors: Optional[List[str]] = None,
    ) -> ProbeResult:
        """
        Probe AI platforms for brand recommendability.

        This is the core method. It sends your URL to multiple AI platforms,
        analyzes responses for brand mentions, and computes a composite score.

        Args:
            url: The URL/brand to probe.
            platforms: AI platforms to check (default: chatgpt, claude, perplexity, gemini).
            prompts: Custom prompts to test. Auto-generated if not provided.
            threshold: Minimum score — result.status = "fail" if below.
            compare_competitors: Competitor URLs to benchmark against.

        Returns:
            ProbeResult with score, platform breakdown, and recommended actions.

        Example:
            >>> result = client.probe("https://myapp.com")
            >>> print(result.score)  # 72.5
            >>> print(result.mentioned_by)  # 3
        """
        payload = {
            "url": url,
            "platforms": platforms or self.DEFAULT_PLATFORMS,
            "threshold": threshold,
        }
        if prompts:
            payload["prompts"] = prompts
        if compare_competitors:
            payload["compare_competitors"] = compare_competitors

        response = self._request("POST", "/probe", payload)
        return self._parse_probe_result(response)

    def score(self, url: str) -> ScoreReport:
        """
        Get current and historical recommendability score.

        Args:
            url: The URL/brand to check.

        Returns:
            ScoreReport with current score, previous score, delta, and trend.

        Example:
            >>> report = client.score("https://myapp.com")
            >>> print(f"Score: {report.current_score} ({report.trend})")
        """
        response = self._request("GET", f"/score?url={url}")
        return ScoreReport(
            brand=response.get("brand", ""),
            current_score=response.get("current_score", 0),
            previous_score=response.get("previous_score"),
            score_delta=response.get("score_delta"),
            trend=response.get("trend", "stable"),
            history=response.get("history", []),
        )

    def simulate(
        self,
        url: str,
        action: str,
        n_trials: int = 1000,
    ) -> SimulationResult:
        """
        Run counterfactual simulation: "If I do X, what happens to my score?"

        Uses Monte Carlo simulation to predict the impact of an action
        on your AI recommendability score, with confidence intervals.

        Args:
            url: The URL/brand to simulate for.
            action: Description of the action to simulate.
            n_trials: Number of Monte Carlo trials (default: 1000).

        Returns:
            SimulationResult with predicted score, uplift, and confidence interval.

        Example:
            >>> sim = client.simulate("https://myapp.com", "Add FAQ schema markup")
            >>> print(f"+{sim.uplift_percent}% (CI: {sim.confidence_lower}-{sim.confidence_upper})")
        """
        payload = {
            "url": url,
            "action": action,
            "n_trials": n_trials,
        }
        response = self._request("POST", "/simulate", payload)
        return SimulationResult(
            action=response.get("action", action),
            current_score=response.get("current_score", 0),
            predicted_score=response.get("predicted_score", 0),
            uplift_percent=response.get("uplift_percent", 0),
            confidence_lower=response.get("confidence_lower", 0),
            confidence_upper=response.get("confidence_upper", 0),
            confidence_level=response.get("confidence_level", 0.95),
            n_trials=response.get("n_trials", n_trials),
            recommendation=response.get("recommendation", ""),
        )

    def actions(self, url: str, top_k: int = 5) -> List[Action]:
        """
        Get top recommended actions for improving AI visibility.

        Args:
            url: The URL/brand to get recommendations for.
            top_k: Number of top actions to return.

        Returns:
            List of Action objects, ordered by predicted impact.

        Example:
            >>> for action in client.actions("https://myapp.com"):
            ...     print(f"{action.action} → {action.predicted_uplift}")
        """
        response = self._request("GET", f"/actions?url={url}&top_k={top_k}")
        return [
            Action(
                action=a.get("action", ""),
                impact=a.get("impact", "medium"),
                predicted_uplift=a.get("predicted_uplift", ""),
                platforms_affected=a.get("platforms_affected", []),
                confidence_interval=a.get("confidence_interval"),
            )
            for a in response.get("actions", [])
        ]

    def citations(self, url: str) -> List[Dict[str, Any]]:
        """
        Get source citations that AI platforms use when discussing your brand.

        Returns which URLs, documents, and sources AI models cite when they
        mention (or fail to mention) your brand.

        Args:
            url: The URL/brand to check citations for.

        Returns:
            List of citation dictionaries with source, platform, and context.
        """
        response = self._request("GET", f"/citations?url={url}")
        return response.get("citations", [])

    def entity_graph(self, url: str) -> Dict[str, Any]:
        """
        Get entity graph analysis for your brand.

        Returns the knowledge graph showing relationships between your brand
        and related entities, including PageRank centrality scores.

        Args:
            url: The URL/brand to analyze.

        Returns:
            Dict with nodes, edges, centrality scores, and graph metrics.
        """
        response = self._request("GET", f"/entity-graph?url={url}")
        return response

    # ─── Internal Methods ────────────────────────────────────────────────

    def _request(
        self, method: str, path: str, payload: Optional[dict] = None
    ) -> dict:
        """Make an authenticated API request with retries."""
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "oculus-sdk-python/0.1.0",
        }

        for attempt in range(self.max_retries):
            try:
                if method == "GET":
                    req = Request(url, headers=headers)
                else:
                    data = json.dumps(payload or {}).encode("utf-8")
                    req = Request(url, data=data, headers=headers, method=method)

                with urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            except HTTPError as e:
                if e.code == 401:
                    raise OculusAuthError("Invalid API key", status_code=401)
                elif e.code == 429:
                    retry_after = int(e.headers.get("Retry-After", 60))
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_after)
                        continue
                    raise OculusRateLimitError(retry_after)
                elif e.code >= 500 and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    body = e.read().decode("utf-8") if e.fp else ""
                    raise OculusError(
                        f"API error: {e.code}", status_code=e.code, response=body
                    )
            except URLError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise OculusError(f"Connection error: {e.reason}")

        raise OculusError("Max retries exceeded")

    def _parse_probe_result(self, data: dict) -> ProbeResult:
        """Parse API response into ProbeResult."""
        platform_results = []
        for pr in data.get("platform_results", []):
            analysis = pr.get("analysis", {})
            platform_results.append(
                PlatformResult(
                    platform=pr.get("platform", ""),
                    platform_name=pr.get("platform_name", pr.get("platform", "")),
                    status=pr.get("status", "error"),
                    mention_count=analysis.get("mention_count", 0),
                    url_mentioned=analysis.get("url_mentioned", False),
                    position_score=analysis.get("position_score", 0),
                    mention_score=analysis.get("mention_score", 0),
                    recommendation_score=analysis.get("recommendation_score", 0),
                    composite=analysis.get("composite", 0),
                    recommended=analysis.get("recommended", False),
                    citations=analysis.get("citations", []),
                    response_text=pr.get("response", ""),
                    error=pr.get("error"),
                    timestamp=pr.get("timestamp", ""),
                )
            )

        actions = [
            Action(
                action=a.get("action", ""),
                impact=a.get("impact", "medium"),
                predicted_uplift=a.get("predicted_uplift", ""),
                platforms_affected=a.get("platforms_affected", []),
            )
            for a in data.get("actions", [])
        ]

        return ProbeResult(
            brand=data.get("brand", ""),
            url=data.get("url", ""),
            composite_score=data.get("composite_score", 0),
            mention_count=data.get("mention_count", 0),
            total_platforms=data.get("total_platforms", 0),
            status=data.get("status", "pass"),
            platform_results=platform_results,
            actions=actions,
            timestamp=data.get("timestamp", ""),
            report_url=data.get("report_url"),
        )

    def __repr__(self):
        masked = self.api_key[:8] + "..." if len(self.api_key) > 8 else "***"
        return f"OculusClient(api_key='{masked}', base_url='{self.base_url}')"
