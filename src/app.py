import os
import json
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict, Any

from rag_engine import HybridRAGEngine, simple_tokenize_vi
from rank_bm25 import BM25Okapi


# =========================
# Highlight renderer
# =========================
def render_highlight_page(source_info: Dict[str, Any], file_map: Dict[str, str], zoom: float = 2.5):
    """
    source_info: {"source_name":..., "page":..., "all_bboxes":...}
    file_map: {source_name -> local_path}
    return: np.ndarray (H,W,3) or None
    """
    if not source_info:
        return None

    source_name = source_info.get("source_name")
    page_num = int(source_info.get("page", 1)) - 1

    bboxes = source_info.get("all_bboxes") or source_info.get("bboxes") or []
    if isinstance(bboxes, str):
        try:
            bboxes = json.loads(bboxes)
        except:
            bboxes = []

    pdf_path = file_map.get(source_name)
    if not pdf_path or not os.path.exists(pdf_path):
        return None

    doc = fitz.open(pdf_path)
    if page_num < 0 or page_num >= len(doc):
        doc.close()
        return None

    page = doc[page_num]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    img = img[:, :, :3].copy()
    H, W = img.shape[:2]

    # --- styling ---
    PAD = 4
    ALPHA = 0.30
    BORDER = 2

    if bboxes:
        yellow = np.array([255, 255, 0], dtype=np.uint8)
        border_color = np.array([220, 50, 50], dtype=np.uint8)

        for box in bboxes:
            try:
                x0, y0, x1, y1 = box
                x0 = int(x0 * zoom) - PAD
                y0 = int(y0 * zoom) - PAD
                x1 = int(x1 * zoom) + PAD
                y1 = int(y1 * zoom) + PAD

                x0 = max(0, min(x0, W - 1))
                x1 = max(0, min(x1, W - 1))
                y0 = max(0, min(y0, H - 1))
                y1 = max(0, min(y1, H - 1))
                if x1 <= x0 or y1 <= y0:
                    continue

                region = img[y0:y1, x0:x1]
                region[:] = ((1 - ALPHA) * region + ALPHA *
                             yellow).astype(np.uint8)

                # border
                t = BORDER
                img[y0:y0+t, x0:x1] = border_color
                img[y1-t:y1, x0:x1] = border_color
                img[y0:y1, x0:x0+t] = border_color
                img[y0:y1, x1-t:x1] = border_color
            except:
                continue

    doc.close()
    return img


# =========================
# App state init
# =========================
def init_state():
    if "engine" not in st.session_state:
        st.session_state.engine = HybridRAGEngine(
            db_path="./chroma_db",
            collection_name="querydoc_v4",
            embedding_model="intfloat/multilingual-e5-small",
            llm_name="qwen2.5:7b",
        )
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "chat" not in st.session_state:
        # streamlit chat: list of {"role": "user"/"assistant", "content": "..."}
        st.session_state.chat = []
    if "file_map" not in st.session_state:
        # {source_name -> saved_path}
        st.session_state.file_map = {}
    if "sources_last" not in st.session_state:
        st.session_state.sources_last = []


def rebuild_multifile_index(engine: HybridRAGEngine, all_passages: List[Dict[str, Any]]):
    engine._passages = all_passages
    engine._bm25 = BM25Okapi(
        [simple_tokenize_vi(p["page_content"]) for p in all_passages])
    engine._key_to_passage = {p["metadata"]
                              ["passage_id"]: p for p in all_passages}
    engine._indexed = True


# =========================
# UI
# =========================
st.set_page_config(page_title="Hybrid PDF RAG", layout="wide")
init_state()

st.title("📄 QueryDoc")
st.caption(
    "Upload PDF → Indexing → Chat → Xem highlight evidence trên trang PDF")

left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.subheader("1) Upload & Index")

    uploaded_files = st.file_uploader(
        "Chọn PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    colA, colB = st.columns(2)
    with colA:
        do_index = st.button("📌 Indexing", use_container_width=True)
    with colB:
        do_reset = st.button("🔄 Reset", use_container_width=True)

    if do_reset:
        # reset everything
        st.session_state.indexed = False
        st.session_state.chat = []
        st.session_state.file_map = {}
        st.session_state.sources_last = []
        # tạo engine mới
        st.session_state.engine = HybridRAGEngine(
            db_path="./chroma_db",
            collection_name="querydoc_v4",
            embedding_model="intfloat/multilingual-e5-small",
            llm_name="qwen2.5:7b",
        )
        st.success("Đã reset.")

    if do_index:
        if not uploaded_files:
            st.warning("Chưa có file PDF.")
        else:
            os.makedirs("./uploaded_pdfs", exist_ok=True)
            engine = st.session_state.engine

            all_passages = []
            logs = []

            for uf in uploaded_files:
                save_path = os.path.join("./uploaded_pdfs", uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())

                # map source_name -> path
                st.session_state.file_map[uf.name] = save_path

                info = engine.index_pdf(save_path)
                if info.get("ok"):
                    logs.append(
                        f"✅ Indexed: {uf.name} (passages={info.get('count')})")
                    all_passages.extend(engine._passages)
                else:
                    logs.append(f"❌ Index fail: {uf.name}")

            if all_passages:
                rebuild_multifile_index(engine, all_passages)
                st.session_state.indexed = True
                st.success(f"Index xong. Tổng passages: {len(all_passages)}")
            else:
                st.session_state.indexed = False
                st.error("Không index được passage nào.")

            st.write("\n".join(logs))


with right:
    st.subheader("2) Chat")
    if not st.session_state.indexed:
        st.info("Hãy upload + bấm Index trước.")
    else:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input(
            "Nhập câu hỏi")
        if user_msg:
            # append user
            st.session_state.chat.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            engine = st.session_state.engine
            result = engine.answer(
                query=user_msg, chat_history=st.session_state.chat)

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            st.session_state.sources_last = sources

            st.session_state.chat.append(
                {"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

            if sources:
                with st.expander("📍 Sources (click để xem)", expanded=True):
                    for i, s in enumerate(sources, start=1):
                        st.markdown(
                            f"**Nguồn {i} — {s.get('source_name')} — Trang {s.get('page')}**")
                        st.write((s.get("content") or "")[
                                 :600] + ("..." if len((s.get("content") or "")) > 600 else ""))
                        st.divider()


# =========================
# Highlight panel (source picker)
# =========================
st.subheader("3) Highlight theo nguồn (chọn 1 nguồn để hiển thị trang)")

sources = st.session_state.sources_last or []
if not sources:
    st.write("Chưa có sources để highlight.")
else:
    labels = [
        f"{i+1}) {s.get('source_name')} — Page {s.get('page')}" for i, s in enumerate(sources)]
    pick = st.selectbox("Chọn nguồn", labels, index=0)

    idx = int(pick.split(")")[0]) - 1
    idx = max(0, min(idx, len(sources) - 1))
    chosen = sources[idx]

    img = render_highlight_page(chosen, st.session_state.file_map, zoom=2.5)
    if img is None:
        st.warning(
            "Không render được highlight (thiếu bbox hoặc không tìm thấy file).")
    else:
        st.image(
            img, caption=f"{chosen.get('source_name')} — Page {chosen.get('page')}", use_container_width=True)
