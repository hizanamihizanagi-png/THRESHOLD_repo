"""
Oculus AI — Python SDK
The Stripe of AI Visibility. Measure and optimize your AI recommendability.

Usage:
    from oculus_sdk import OculusClient

    client = OculusClient(api_key="your-key")
    result = client.probe("https://yoursite.com", platforms=["chatgpt", "claude"])
    print(result.score)
"""

from oculus_sdk.client import OculusClient
from oculus_sdk.models import (
    ProbeResult,
    PlatformResult,
    ScoreReport,
    Action,
    SimulationResult,
)

__version__ = "0.1.0"
__all__ = [
    "OculusClient",
    "ProbeResult",
    "PlatformResult",
    "ScoreReport",
    "Action",
    "SimulationResult",
]
