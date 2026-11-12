# Diseño del sistema de recuperación

## Índice administrado

El índice `incident-knowledge-cloud` se despliega como Pinecone Serverless en AWS `us-east-1`.
Su campo vectorial `embedding` utiliza dimensión 1536 y similitud coseno, parámetros compatibles
con `text-embedding-3-small`. Los registros se aíslan en el namespace `technical-docs`.

Cada fragmento conserva el texto original y los metadatos `document_id`, `source`, `page`,
`category`, `tags` y `chunk_index`. El identificador del fragmento se calcula con SHA-256 a partir
del documento, la página, la posición y el contenido, por lo que una nueva ingesta del mismo corpus
actualiza los registros existentes.

## Ingesta

El cargador acepta archivos Markdown, texto, JSON y PDF. Los documentos se dividen con
`RecursiveCharacterTextSplitter` en fragmentos de 600 tokens léxicos y 100 tokens de solapamiento.
Los embeddings se calculan en forma asíncrona y se cargan en lotes de hasta 100 registros.

## Recuperación híbrida

`RAGSystem` combina dos recuperadores mediante Reciprocal Rank Fusion:

- Pinecone aporta similitud semántica con peso 0,60.
- BM25 aporta coincidencia léxica con peso 0,40.

Cada recuperador genera hasta 15 candidatos. La fusión elimina documentos duplicados y retorna los
cinco documentos mejor posicionados. BM25 se construye desde el mismo corpus que se carga en la
nube, de modo que ambas ramas comparten contenido y metadatos.

## Evaluación

El Golden Set contiene cinco preguntas y sus identificadores de documento relevantes. Para cada
pregunta se toman los primeros cinco resultados:

`Precision@5 = documentos relevantes recuperados / 5`

`Recall@5 = documentos relevantes recuperados / documentos relevantes esperados`

El reporte incluye los resultados por consulta y los promedios del conjunto completo.
