from ingest import process_pdf
import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorDB:
    def __init__(
        self,
        collection_name: str = "querydoc_v4",
        persist_dir: str = "./chroma_db",
        model_name: str = "intfloat/multilingual-e5-small"
    ):
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            persist_directory=persist_dir
        )

    def add_passages(self, passages):
        # Model E5: Luôn cần 'passage: ' khi nạp
        texts = ["passage: " + p["page_content"] for p in passages]
        metadatas = []
        ids = []

        for p in passages:
            meta = p["metadata"].copy()
            if "all_bboxes" in meta:
                meta["all_bboxes"] = json.dumps(meta["all_bboxes"])
            metadatas.append(meta)

            unique_id = meta.get(
                "passage_id") or f"{meta['doc_id']}_{meta.get('start_line_id', 'na')}"
            ids.append(unique_id)

        self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def retrieve(self, query: str, k: int = 4, doc_id: str = None, use_mmr: bool = True):
        """
        Wrapper thay thế hoàn toàn cho get_retriever().
        Đảm bảo 100% query được prefix 'query: '
        """
        query_ef = "query: " + query

        search_kwargs = {"k": k}
        if doc_id:
            search_kwargs["filter"] = {"doc_id": doc_id}

        if use_mmr:
            # MMR giúp đa dạng hóa kết quả (rất tốt cho highlight nhiều trang)
            docs = self.db.max_marginal_relevance_search(
                query_ef, fetch_k=20, lambda_mult=0.5, **search_kwargs
            )
        else:
            docs = self.db.similarity_search(query_ef, **search_kwargs)

        # Hậu xử lý: Giải nén bboxes để UI dùng được ngay
        for doc in docs:
            if "all_bboxes" in doc.metadata:
                try:
                    doc.metadata["all_bboxes"] = json.loads(
                        doc.metadata["all_bboxes"])
                except:
                    pass
        return docs

    def search_with_score(self, query: str, k: int = 4, doc_id: str = None):
        """Dùng để debug hoặc khi cần biết Confidence Score"""
        query_ef = "query: " + query
        filter_dict = {"doc_id": doc_id} if doc_id else None

        results = self.db.similarity_search_with_score(
            query_ef, k=k, filter=filter_dict)

        for doc, score in results:
            if "all_bboxes" in doc.metadata:
                doc.metadata["all_bboxes"] = json.loads(
                    doc.metadata["all_bboxes"])
        return results
