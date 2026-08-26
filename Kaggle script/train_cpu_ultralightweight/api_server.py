"""
MÁY CHỦ HẬU XỬ LÝ FASTAPI (POST-PROCESSING SERVER - Phụ lục C)
Cung cấp API /api/compose, /api/segment, /api/suggest, /api/health
"""

import os
from typing import List, Optional, Set
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from asl_pipeline import ASLStreamPipeline


app = FastAPI(
    title="ASL Fingerspelling to Sentence Pipeline API",
    description="Backend hậu xử lý chuỗi ký tự ASL thành câu tiếng Anh và dịch sang tiếng Việt.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = ASLStreamPipeline()


class ComposeRequest(BaseModel):
    words: List[str]                      # Danh sách từ thô: ["HELO", "MY", "NAM", "IS", "ADLEY"]
    mode: str = "fast"                    # "fast" = SymSpell + ASL Rules | "llm" = Large Language Model
    target_lang: Optional[str] = "vi"     # "vi" = Dịch sang tiếng Việt
    lexicon: List[str] = []               # Danh sách tên riêng / whitelist theo phiên
    speaker_pronoun: Optional[str] = "tôi"
    listener_pronoun: Optional[str] = "bạn"


class ComposeResponse(BaseModel):
    raw: str
    english: str
    translated: Optional[str] = None
    confidence: float
    engine: str


class SegmentRequest(BaseModel):
    blob: str                             # Chuỗi dính liền không dấu cách: "whereisthehospital"


class SuggestResponse(BaseModel):
    prefix: str
    suggestions: List[str]


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "ASL Post-Processing Pipeline"}


@app.post("/api/compose", response_model=ComposeResponse)
def compose_sentence(req: ComposeRequest):
    """
    Nhận danh sách từ thô từ trình duyệt, sửa lỗi chính tả, dựng câu và dịch sang tiếng Việt.
    """
    if req.speaker_pronoun and req.listener_pronoun:
        pipeline.translator.set_pronouns(req.speaker_pronoun, req.listener_pronoun)

    res = pipeline.process_raw_word_list(req.words, custom_lexicon=req.lexicon)
    return ComposeResponse(
        raw=res["raw"],
        english=res["english"],
        translated=res["translated"] if req.target_lang == "vi" else None,
        confidence=res["confidence"],
        engine=res["engine"]
    )


@app.post("/api/segment")
def segment_unspaced_text(req: SegmentRequest):
    """
    Tách từ cho chuỗi dính liền không có dấu cách.
    """
    res = pipeline.process_unspaced_blob(req.blob)
    return res


@app.get("/api/suggest", response_model=SuggestResponse)
def get_suggestions(prefix: str = Query(..., min_length=1)):
    """
    Gợi ý Top-3 từ thông minh theo tiền tố khi người dùng đang đánh vần.
    """
    suggs = pipeline.trie.suggest_completions(prefix, top_k=3)
    return SuggestResponse(
        prefix=prefix,
        suggestions=[w for w, _ in suggs]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
