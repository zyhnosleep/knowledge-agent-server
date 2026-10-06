"""Start the project API plus an opt-in multimodal query endpoint."""
from typing import Literal

import uvicorn
from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.dependencies import require_business_api_user
from app.db.session import get_db
from app.main import app
from app.services.multimodal_rag import MultimodalRAG


class Request(BaseModel):
    project_slug: str = "multimodal-pilot"
    question: str = Field(min_length=1, max_length=6000)
    mode: Literal["text", "text_image"] = "text_image"
    max_pages: int = Field(default=3, ge=1, le=3)


@app.post("/api/multimodal/query", dependencies=[Depends(require_business_api_user)])
def query(request: Request, db: Session = Depends(get_db)):
    service = MultimodalRAG(db)
    try:
        retrieval = service.retrieve(request.project_slug, request.question, request.max_pages)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"retrieval": retrieval, "generation": service.generate(retrieval, request.mode)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=18002)
