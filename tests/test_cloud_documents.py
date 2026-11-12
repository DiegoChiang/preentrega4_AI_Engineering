import json

from incident_intelligence.rag.chunking import count_lexical_tokens, split_documents
from incident_intelligence.rag.documents import load_knowledge_documents


def test_loads_markdown_and_json_with_metadata(tmp_path):
    (tmp_path / "manual_red.md").write_text("Diagnóstico de red", encoding="utf-8")
    (tmp_path / "records.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "obs-1",
                        "content": "Métricas y trazas",
                        "page": 3,
                        "category": "observabilidad",
                        "tags": ["SLO", "slo", "trazas"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    documents = load_knowledge_documents(tmp_path)

    assert len(documents) == 2
    json_document = next(item for item in documents if item.document_id == "obs-1")
    assert json_document.source == "records.json"
    assert json_document.page == 3
    assert json_document.category == "observabilidad"
    assert json_document.tags == ["slo", "trazas"]


def test_split_documents_has_stable_ids_and_bounded_size(tmp_path):
    text = " ".join(f"token{index}" for index in range(1_200))
    (tmp_path / "manual.md").write_text(text, encoding="utf-8")
    documents = load_knowledge_documents(tmp_path)

    first = split_documents(documents, chunk_size=600, chunk_overlap=100)
    second = split_documents(documents, chunk_size=600, chunk_overlap=100)

    assert len(first) >= 2
    assert [item.id for item in first] == [item.id for item in second]
    assert all(count_lexical_tokens(item.content) <= 600 for item in first)
    assert all(item.source == "manual.md" for item in first)
