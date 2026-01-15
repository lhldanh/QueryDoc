import os
import re
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from ingest import process_pdf
from vector_store import VectorDB


# --- Tokenizer ---


def simple_tokenize_vi(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

# --- RRF Fusion ---


def rrf_fuse(ranked_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    scores = defaultdict(float)
    for lst in ranked_lists:
        for rank, key in enumerate(lst, start=1):
            scores[key] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRAGEngine:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "querydoc_v4",
        embedding_model: str = "intfloat/multilingual-e5-small",
        llm_name: str = "qwen2.5:7b",
    ):
        self.vdb = VectorDB(
            persist_dir=db_path,
            collection_name=collection_name,
            model_name=embedding_model,
        )
        self.llm = Ollama(model=llm_name)

        template = """Bạn là chuyên gia phân tích dữ liệu. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên CONTEXT và LỊCH SỬ trò chuyện.

LỊCH SỬ TRÒ CHUYỆN:
{chat_history}

CONTEXT:
{context}

---
CÂU HỎI: {question}

RÀO CHẮN:
- Nếu thông tin không có trong CONTEXT, hãy trả lời: "Tôi rất tiếc, tài liệu không cung cấp thông tin này."
- Phải trích dẫn chính xác số thứ tự đoạn chứa thông tin dùng để trả lời theo dạng [Đoạn x].
- Trình bày câu trả lời ngắn gọn và format ở sau "TRẢ LỜI:"

TRẢ LỜI:
(Nội dung trả lời)

Trích dẫn từ CONTEXT: [Đoạn x], [Đoạn y]"""

        self.prompt = PromptTemplate.from_template(template)
        self._passages = []
        self._bm25 = None
        self._key_to_passage = {}
        self._indexed = False

    def index_pdf(self, pdf_path: str):
        passages = process_pdf(pdf_path)
        if not passages:
            return {"ok": False}

        for p in passages:
            meta = p["metadata"]
            meta["source_name"] = meta.get(
                "source_name", os.path.basename(pdf_path))
            meta["passage_id"] = meta.get(
                "passage_id", f"{meta.get('doc_id', 'na')}_p{meta.get('page', 'na')}_{meta.get('start_line_id', 'na')}")

        self.vdb.add_passages(passages)
        self._passages = passages
        self._bm25 = BM25Okapi(
            [simple_tokenize_vi(p["page_content"]) for p in passages])
        self._key_to_passage = {p["metadata"]
                                ["passage_id"]: p for p in passages}
        self._indexed = True
        return {"ok": True, "count": len(passages)}

    def hybrid_retrieve(self, query: str, topk: int = 6, bm25_k: int = 30, vec_k: int = 20, doc_id: str = None) -> List[Dict]:
        if not self._indexed:
            raise RuntimeError("Vui lòng index tài liệu trước!")

        q_tokens = simple_tokenize_vi(query)
        bm_scores = self._bm25.get_scores(q_tokens)
        bm_idxs = sorted(range(len(bm_scores)),
                         key=lambda i: bm_scores[i], reverse=True)[:bm25_k]
        bm25_keys = [self._passages[i]["metadata"]["passage_id"]
                     for i in bm_idxs]

        vec_docs = self.vdb.retrieve(
            query, k=vec_k, doc_id=doc_id, use_mmr=False)
        vec_keys = [d.metadata.get("passage_id") for d in vec_docs]

        fused = rrf_fuse([bm25_keys, vec_keys])

        final_passages = []
        for pid, _ in fused:
            if pid in self._key_to_passage:
                final_passages.append(self._key_to_passage[pid])
            if len(final_passages) >= topk:
                break
        return final_passages

    def answer(
        self,
        query: str,
        chat_history: List[Dict] = None,
        topk: int = 6,
        bm25_k: int = 30,
        vec_k: int = 20,
        doc_id: str = None
    ) -> Dict:
        history_str = ""
        if chat_history:
            # Lấy tối đa 3 lượt gần nhất để tránh tràn context
            for turn in chat_history[-3:]:
                role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
                history_str += f"{role}: {turn['content']}\n"
        else:
            history_str = "Không có lịch sử."

        passages = self.hybrid_retrieve(
            query, topk=topk, bm25_k=bm25_k, vec_k=vec_k, doc_id=doc_id)
        if not passages:
            return {"answer": "Không tìm thấy thông tin.", "sources": []}

        context_text = "\n\n".join(
            [f"[Đoạn {i+1}]: {p['page_content']}" for i, p in enumerate(passages)])

        ans_content = self.llm.invoke(self.prompt.format(
            chat_history=history_str,
            context=context_text,
            question=query
        ))

        ans_text = getattr(ans_content, 'content', str(ans_content))

        used_indices = re.findall(r'\[(?:Đoạn\s*)?(\d+)\]', ans_text)
        used_indices = set(int(idx) - 1 for idx in used_indices)

        clean_answer = ans_text
        patterns_to_remove = [
            r'(?i)Trích dẫn từ CONTEXT:',
            r'(?i)^TRẢ LỜI:',
            r'\[(?:Đoạn\s*)?\d+\]'
        ]

        for pattern in patterns_to_remove:
            clean_answer = re.sub(pattern, '', clean_answer)

        lines = [line.strip()
                 for line in clean_answer.split('\n') if line.strip()]
        clean_answer = "\n\n".join(lines)

        final_sources = []
        for i, p in enumerate(passages):
            if i in used_indices:
                bboxes = p["metadata"].get("all_bboxes", [])
                if isinstance(bboxes, str):
                    try:
                        bboxes = json.loads(bboxes)
                    except:
                        bboxes = []

                final_sources.append({
                    "id": i + 1,
                    "content": p["page_content"],
                    "page": p["metadata"].get("page", -1),
                    "source_name": p["metadata"].get("source_name"),
                    "bboxes": bboxes,
                })

        return {
            "answer": clean_answer if clean_answer else ans_text,
            "sources": final_sources
        }
