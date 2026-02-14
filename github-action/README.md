# 🔮 Oculus AI — Recommendability Check

**The Stripe of AI Visibility.** Check your AI recommendability score in every deploy.

[![GitHub Action](https://img.shields.io/badge/GitHub%20Action-Oculus%20AI-purple?logo=github)](https://github.com/hizanamihizanagi-png/THRESHOLD_repo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What it does

This GitHub Action probes AI platforms (ChatGPT, Claude, Perplexity, Gemini, Copilot, Mistral, DeepSeek) to measure how likely they are to **recommend your brand**. It runs in your CI/CD pipeline and fails builds if your AI visibility drops.

**Think Lighthouse for performance → Oculus for AI recommendability.**

## Quick Start

```yaml
name: AI Recommendability Check
on: [push, pull_request]

jobs:
  check-ai-visibility:
    runs-on: ubuntu-latest
    steps:
      - name: Oculus AI Recommendability Check
        uses: hizanamihizanagi-png/THRESHOLD_repo/github-action@main
        with:
          api_key: ${{ secrets.OCULUS_API_KEY }}
          url: 'https://yoursite.com'
          platforms: 'chatgpt,claude,perplexity,gemini'
          threshold: 50
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          PERPLEXITY_API_KEY: ${{ secrets.PERPLEXITY_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
```

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `api_key` | ✅ | — | Your Oculus AI API key |
| `url` | ✅ | — | URL to probe for AI recommendability |
| `prompts` | ❌ | Auto-generated | Comma-separated prompts to test |
| `platforms` | ❌ | `chatgpt,claude,perplexity,gemini` | AI platforms to probe |
| `threshold` | ❌ | `0` | Minimum score (0-100). Build fails if below. |
| `fail_on_drop` | ❌ | `0` | Fail if score dropped by this % |
| `compare_competitors` | ❌ | — | Competitor URLs to benchmark against |
| `output_format` | ❌ | `summary` | Output: `summary`, `json`, or `markdown` |

## Outputs

| Output | Description |
|--------|-------------|
| `score` | Composite recommendability score (0-100) |
| `previous_score` | Score from last run |
| `score_delta` | Change from previous |
| `mention_count` | Platforms that mentioned your brand |
| `total_platforms` | Total platforms probed |
| `top_actions` | JSON array of top 3 recommended actions |
| `status` | `pass` or `fail` |

## Supported Platforms

| Platform | API Key Variable | Status |
|----------|-----------------|--------|
| ChatGPT (OpenAI) | `OPENAI_API_KEY` | ✅ |
| Claude (Anthropic) | `ANTHROPIC_API_KEY` | ✅ |
| Perplexity AI | `PERPLEXITY_API_KEY` | ✅ |
| Google Gemini | `GOOGLE_API_KEY` | ✅ |
| Microsoft Copilot | `AZURE_API_KEY` | ✅ |
| Mistral AI | `MISTRAL_API_KEY` | ✅ |
| DeepSeek | `DEEPSEEK_API_KEY` | ✅ |

## How Scoring Works

Each AI platform response is analyzed for:

- **Mention Score** (30%) — How often your brand is mentioned
- **Position Score** (25%) — How early in the response you appear
- **Recommendation Score** (30%) — Whether AI uses recommendation language
- **Citation Score** (15%) — Whether your URL is cited as a source

The composite score (0-100) represents the probability that AI assistants will recommend your company.

## Why Oculus?

| Feature | MentionDesk | Generic SEO | **Oculus** |
|---------|-------------|-------------|-----------|
| Track AI mentions | ✅ | ❌ | ✅ |
| CI/CD integration | ❌ | ❌ | **✅** |
| Entity graph analysis | ❌ | ❌ | **✅** |
| Predictive simulator | ❌ | ❌ | **✅** |
| API-first | ❌ | ❌ | **✅** |
| Confidence intervals | ❌ | ❌ | **✅** |

---

**Stop optimizing for clicks. Start optimizing for LLMs.**

[Get your API key →](https://oculus-ai.dev)
