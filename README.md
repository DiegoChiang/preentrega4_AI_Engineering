# Sistema RAG híbrido en Pinecone

Sistema de recuperación de documentación técnica que combina búsqueda semántica en Pinecone
Serverless con búsqueda léxica BM25. Procesa Markdown, texto, JSON y PDF; conserva metadatos de
origen; y mide Precision@5 y Recall@5 sobre un Golden Set.

## Componentes

- Índice serverless con vectores de 1536 dimensiones y similitud coseno.
- Embeddings `text-embedding-3-small`.
- Fragmentos de 600 tokens con solapamiento de 100.
- Metadatos: fuente, página, categoría, etiquetas, documento y posición del fragmento.
- Recuperación híbrida con pesos 0,60 semántico y 0,40 BM25.
- Cinco resultados únicos por consulta.
- Benchmark reproducible de cinco preguntas.

## Instalación

Requiere Python 3.12 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
copy .env.example .env
```

Configurar las credenciales en `.env`:

```dotenv
OPENAI_API_KEY=...
PINECONE_API_KEY=...
```

El archivo `.env` está excluido del control de versiones.

## Uso

Crear o validar el índice:

```bash
cloud-rag setup
```

Procesar el corpus y cargarlo en Pinecone:

```bash
python ingest.py
```

Ejecutar una consulta híbrida:

```bash
cloud-rag query "¿Qué revisar cuando la API de pagos presenta errores 5xx?"
```

Calcular las métricas del Golden Set:

```bash
python evaluate.py
```

También se encuentran disponibles `cloud-rag ingest` y `cloud-rag evaluate`.

## Replicación del índice

La configuración predeterminada crea `incident-knowledge-cloud` en AWS `us-east-1` con un campo
`embedding` de tipo `dense_vector`, dimensión 1536 y métrica coseno. Puede cambiarse mediante las
variables `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `PINECONE_CLOUD` y `PINECONE_REGION`.

La ingesta almacena el texto original junto con `document_id`, `source`, `page`, `category`, `tags`
y `chunk_index`. El namespace evita mezclar el corpus técnico con datos de otras aplicaciones.

## Estructura

```text
data/                         corpus técnico
evaluation/golden_set.json   preguntas y documentos esperados
src/incident_intelligence/
  rag/documents.py            carga multiformato
  rag/chunking.py             fragmentación reproducible
  rag/pinecone.py             índice serverless
  rag/ingestion.py            embeddings y carga por lotes
  rag/retrieval.py            Pinecone + BM25
  rag/evaluation.py           Precision@5 y Recall@5
ingest.py                     ejecutable de ingesta
evaluate.py                   ejecutable de evaluación
```

La descripción de arquitectura y métricas está en [docs/cloud_rag.md](docs/cloud_rag.md).

## Calidad

```bash
pytest
ruff check .
```

Las pruebas cubren carga y metadatos, fragmentación, configuración del índice, ingesta por lotes,
fusión de resultados y cálculo del benchmark sin depender de recursos externos.
