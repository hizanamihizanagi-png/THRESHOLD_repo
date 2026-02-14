# 🔮 Oculus AI — Python SDK

**The Stripe of AI Visibility.** Measure and optimize your AI recommendability with one line of code.

```bash
pip install oculus-ai
```

## Quick Start (60 seconds)

```python
from oculus_sdk import OculusClient

client = OculusClient(api_key="ocu_live_your_key_here")

# Check your AI recommendability
result = client.probe("https://yoursite.com")
print(f"Score: {result.score}/100")
print(f"Mentioned by: {result.mentioned_by}/{result.total_platforms} AI platforms")

# See what to fix
for action in result.actions:
    print(f"→ {action.action} (predicted uplift: {action.predicted_uplift})")
```

## Core Methods

### `client.probe(url)` — Check AI Recommendability
```python
result = client.probe(
    url="https://yoursite.com",
    platforms=["chatgpt", "claude", "perplexity", "gemini"],
    prompts=["best CRM for startups"],
    threshold=50,  # fail if score below 50
)

# result.score → 72.5
# result.status → "pass"
# result.mentioned_by → 3
# result.platform_results → [PlatformResult, ...]
# result.actions → [Action, ...]
```

### `client.simulate(url, action)` — Predict Impact
```python
sim = client.simulate(
    url="https://yoursite.com",
    action="Add FAQ schema markup to all product pages",
)
print(f"Current: {sim.current_score}")
print(f"Predicted: {sim.predicted_score}")
print(f"Uplift: +{sim.uplift_percent}%")
print(f"95% CI: [{sim.confidence_lower}, {sim.confidence_upper}]")
```

### `client.score(url)` — Historical Trends
```python
report = client.score("https://yoursite.com")
print(f"Current: {report.current_score}")
print(f"Previous: {report.previous_score}")
print(f"Trend: {report.trend}")  # "up", "down", "stable"
```

### `client.citations(url)` — Source Intelligence
```python
citations = client.citations("https://yoursite.com")
for c in citations:
    print(f"AI cites {c['source']} on {c['platform']}")
```

### `client.entity_graph(url)` — Knowledge Graph
```python
graph = client.entity_graph("https://yoursite.com")
print(f"Centrality: {graph['centrality_score']}")
print(f"Entities: {len(graph['nodes'])}")
```

## CI/CD Integration

```python
# In your deploy script
import sys
import os
from oculus_sdk import OculusClient

client = OculusClient(api_key=os.environ["OCULUS_API_KEY"])
result = client.probe("https://yoursite.com", threshold=50)

if result.status == "fail":
    print(f"❌ AI visibility dropped to {result.score}/100!")
    for action in result.actions:
        print(f"  Fix: {action.action}")
    sys.exit(1)

print(f"✅ AI visibility: {result.score}/100")
```

## Error Handling

```python
from oculus_sdk import OculusClient
from oculus_sdk.client import OculusAuthError, OculusRateLimitError, OculusError

try:
    result = client.probe("https://yoursite.com")
except OculusAuthError:
    print("Invalid API key")
except OculusRateLimitError as e:
    print(f"Rate limited. Retry in {e.retry_after}s")
except OculusError as e:
    print(f"API error: {e.message}")
```

## Supported AI Platforms

| Platform | Key |
|----------|-----|
| ChatGPT (OpenAI) | `chatgpt` |
| Claude (Anthropic) | `claude` |
| Perplexity AI | `perplexity` |
| Google Gemini | `gemini` |
| Microsoft Copilot | `copilot` |
| Mistral AI | `mistral` |
| DeepSeek | `deepseek` |

---

**Stop optimizing for clicks. Start optimizing for LLMs.**

[Get your API key →](https://oculus-ai.dev) | [GitHub Action →](https://github.com/hizanamihizanagi-png/THRESHOLD_repo/tree/main/github-action) | [Docs →](https://docs.oculus-ai.dev)
