import fitz
import hashlib
import os
import re
import unicodedata
from collections import defaultdict


def get_doc_id(pdf_path):
    stats = os.stat(pdf_path)
    base_info = f"{os.path.basename(pdf_path)}_{stats.st_size}_{stats.st_mtime}"
    return hashlib.md5(base_info.encode()).hexdigest()


def normalize_for_store(text):
    """Giữ nguyên hoa/thường, chuẩn hóa NFC và khoảng trắng để Embedding/Display"""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_for_compare(text):
    """Chuẩn hóa để so sánh (detect boilerplate): NFC + Whitespace + Casefold"""
    return normalize_for_store(text).casefold()


def is_meaningful_text(text):
    return bool(re.search(r'[^\W_]', text, flags=re.UNICODE)) and len(text.strip()) >= 2


def extract_boilerplate(doc, margin_ratio=0.08):
    """Tìm boilerplate dựa trên Page Coverage với chuẩn hóa Casefold"""
    text_to_pages = defaultdict(set)

    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                y_center = (line["bbox"][1] + line["bbox"][3]) / 2
                if y_center < page_height * margin_ratio or y_center > page_height * (1 - margin_ratio):
                    raw_text = "".join([s["text"] for s in line["spans"]])
                    norm_text = normalize_for_compare(raw_text)
                    if is_meaningful_text(norm_text):
                        text_to_pages[norm_text].add(page_num)

    total_pages = len(doc)
    return {txt for txt, pages in text_to_pages.items()
            if len(pages) >= max(2, total_pages * 0.6)}


def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    doc_id = get_doc_id(pdf_path)
    boilerplate_set = extract_boilerplate(doc)
    all_passages = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_lines = []
        for b_idx, block in enumerate(page_dict["blocks"]):
            if "lines" not in block:
                continue
            for l_idx, line in enumerate(block["lines"]):
                raw_text = "".join([s["text"] for s in line["spans"]])

                # Normalize để lưu trữ (giữ Case)
                display_text = normalize_for_store(raw_text)
                # Normalize để so sánh boilerplate
                compare_text = normalize_for_compare(raw_text)

                if not is_meaningful_text(display_text):
                    continue
                if compare_text in boilerplate_set:
                    continue

                page_lines.append({
                    "text": display_text,
                    "bbox": list(line["bbox"]),
                    "line_id": f"p{page_num}_b{b_idx}_l{l_idx}"
                })

        # Grouping Passages (Windowing)
        window_size = 5
        stride = 3

        for i in range(0, len(page_lines), stride):
            window = page_lines[i: i + window_size]
            if len(window) < 2 and len(page_lines) > 1:
                continue
            if not window:
                break

            # Logic nối dòng (Hyphen-Join) với Unicode Letter check
            combined_text = ""
            for item in window:
                curr = item["text"]
                if combined_text.endswith('-') and re.match(r'^[^\W\d_]', curr, re.UNICODE):
                    combined_text = combined_text[:-1] + curr
                else:
                    combined_text = (combined_text + " " + curr).strip()

            if len(combined_text) < 40:
                continue

            all_passages.append({
                "page_content": combined_text,
                "metadata": {
                    "doc_id": doc_id,
                    # citations
                    "source_path": pdf_path,
                    "source_name": os.path.basename(pdf_path),
                    # trace
                    "page": page_num + 1,
                    "start_line_id": window[0]["line_id"],
                    "end_line_id": window[-1]["line_id"],
                    # highlight
                    "all_bboxes": [d["bbox"] for d in window],
                    # future-proof
                    "content_type": "text",
                    "passage_id": f'{doc_id}_p{page_num+1}_{window[0]["line_id"]}'
                }
            })

    doc.close()
    return all_passages
