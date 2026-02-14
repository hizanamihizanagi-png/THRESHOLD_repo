FROM python:3.12-slim

LABEL maintainer="Oculus AI <hello@oculus-ai.dev>"
LABEL description="Oculus AI Recommendability Check — CI/CD integration for AI visibility"

COPY probe.py /probe.py

RUN chmod +x /probe.py

ENTRYPOINT ["python3", "/probe.py"]
