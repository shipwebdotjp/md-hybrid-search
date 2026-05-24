from sentence_transformers import CrossEncoder
import config

class ReRanker:
    def __init__(self, model_name="cl-nagoya/ruri-reranker-large"):
        path_to_model = str(config.BASE_DIR / "path_to_model" / model_name)
        self.model = CrossEncoder(path_to_model)

    def rerank(self, query, documents, top_k=5):
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
