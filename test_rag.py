"""
Tests for Qdrant RAG and Embeddings
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.knowledge_base.qdrant_rag import qdrant_rag
from app.knowledge_base.embeddings import agri_embedder

def test_embeddings_generation():
    vec = agri_embedder.embed_text("peeli kungi yellow rust gandum")
    assert len(vec) == 384
    assert any(x != 0 for x in vec)

def test_qdrant_retrieval_yellow_rust():
    results = qdrant_rag.search_guidelines("Meri wheat ki fasal par peela haldi powder lag raha hai kungi", limit=3)
    assert len(results) > 0
    top = results[0]["payload"]
    assert "Yellow Rust" in top["condition_name"] or "Rust" in top["condition_name"]
    assert "Puccinia striiformis" in top["scientific_name"]
    assert len(top["official_recommendations"]) > 0

def test_qdrant_retrieval_nitrogen_deficiency():
    results = qdrant_rag.search_guidelines("puraney neechay walay pattay peelay hain urea ki kami", limit=3)
    assert len(results) > 0
    top = results[0]["payload"]
    assert "Nitrogen" in top["condition_name"]

if __name__ == "__main__":
    test_embeddings_generation()
    test_qdrant_retrieval_yellow_rust()
    test_qdrant_retrieval_nitrogen_deficiency()
    print("[PASS] All RAG tests passed successfully!")
