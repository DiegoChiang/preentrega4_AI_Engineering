"""Consulta de ejemplo contra el recuperador híbrido."""

import asyncio
import json

from incident_intelligence.cli.cloud_rag import query_documents
from incident_intelligence.config import get_settings


async def run() -> None:
    results = await query_documents(
        get_settings(),
        "¿Qué revisar cuando la API de pagos presenta errores 5xx?",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
