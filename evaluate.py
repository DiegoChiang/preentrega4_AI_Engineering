"""Ejecuta el benchmark de recuperación Top-5."""

import asyncio
import json

from incident_intelligence.cli.cloud_rag import evaluate_documents
from incident_intelligence.config import get_settings

if __name__ == "__main__":
    result = asyncio.run(evaluate_documents(get_settings()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
