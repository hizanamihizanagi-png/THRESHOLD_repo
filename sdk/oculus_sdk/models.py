"""
Oculus AI SDK — Data Models
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class PlatformResult:
    """Result from probing a single AI platform."""
    platform: str
    platform_name: str
    status: str  # success, skipped, error
    mention_count: int = 0
    url_mentioned: bool = False
    position_score: float = 0.0
    mention_score: float = 0.0
    recommendation_score: float = 0.0
    composite: float = 0.0
    recommended: bool = False
    citations: List[str] = field(default_factory=list)
    response_text: str = ""
    error: Optional[str] = None
    timestamp: str = ""


@dataclass
class Action:
    """Recommended action to improve AI visibility."""
    action: str
    impact: str  # high, medium, low
    predicted_uplift: str
    platforms_affected: List[str] = field(default_factory=list)
    confidence_interval: Optional[str] = None


@dataclass
class ProbeResult:
    """Complete result from an AI recommendability probe."""
    brand: str
    url: str
    composite_score: float
    mention_count: int
    total_platforms: int
    status: str  # pass, fail
    platform_results: List[PlatformResult] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    timestamp: str = ""
    report_url: Optional[str] = None

    @property
    def score(self) -> float:
        """Alias for composite_score."""
        return self.composite_score

    @property
    def mentioned_by(self) -> int:
        """Number of platforms that mentioned the brand."""
        return self.mention_count

    @property
    def is_recommended(self) -> bool:
        """Whether AI platforms are likely to recommend this brand."""
        return self.composite_score >= 50

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "brand": self.brand,
            "url": self.url,
            "composite_score": self.composite_score,
            "mention_count": self.mention_count,
            "total_platforms": self.total_platforms,
            "status": self.status,
            "platform_results": [
                {
                    "platform": r.platform,
                    "status": r.status,
                    "composite": r.composite,
                    "mention_count": r.mention_count,
                    "recommended": r.recommended,
                }
                for r in self.platform_results
            ],
            "actions": [
                {"action": a.action, "impact": a.impact, "uplift": a.predicted_uplift}
                for a in self.actions
            ],
            "timestamp": self.timestamp,
        }


@dataclass
class ScoreReport:
    """Historical score report with trend data."""
    brand: str
    current_score: float
    previous_score: Optional[float] = None
    score_delta: Optional[float] = None
    trend: str = "stable"  # up, down, stable
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result from counterfactual simulation."""
    action: str
    current_score: float
    predicted_score: float
    uplift_percent: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float = 0.95
    n_trials: int = 1000
    recommendation: str = ""
