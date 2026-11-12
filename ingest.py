"""Carga el corpus técnico en Pinecone Serverless."""

import asyncio
import json

from incident_intelligence.cli.cloud_rag import ingest_documents
from incident_intelligence.config import get_settings

if __name__ == "__main__":
    result = asyncio.run(ingest_documents(get_settings()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
