"""Single-GPU, loopback-only Transformers backend for the multimodal pilot.

Run in the existing CUDA environment, independently of the application venv.
The embedding endpoint implements the project's existing Ollama wire format;
the generator is explicitly Qwen3-VL, with no Ollama dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Qwen local multimodal backend")
lock = threading.Lock()
embedding_model = embedding_tokenizer = generator = processor = None
config = None
EMBEDDING_ID = "Qwen/Qwen3-Embedding-4B"
GENERATOR_ID = "Qwen/Qwen3-VL-8B-Instruct"


class EmbedRequest(BaseModel):
    model: str
    input: list[str] = Field(min_length=1, max_length=10000)


class GenerateRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = Field(default=512, ge=1, le=1024)


def event(stage, **fields):
    print(json.dumps({"stage": stage, "time": time.time(), **fields}), flush=True)


def load_embedding():
    global embedding_model, embedding_tokenizer
    if embedding_model is not None:
        return
    from modelscope import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    event("embedding_download", model=EMBEDDING_ID)
    path = snapshot_download(EMBEDDING_ID, cache_dir=str(config.cache),
                             ignore_file_pattern=["*.bin", "*.gguf", "*.pt"])
    embedding_tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    embedding_model = AutoModel.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=False).to("cuda").eval()
    event("embedding_loaded", model=EMBEDDING_ID)


def load_generator():
    global generator, processor
    if generator is not None:
        return
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    event("generator_loading", model=GENERATOR_ID)
    processor = AutoProcessor.from_pretrained(
        config.generator_path, min_pixels=200704, max_pixels=802816,
        trust_remote_code=False)
    generator = Qwen3VLForConditionalGeneration.from_pretrained(
        config.generator_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False).to("cuda").eval()
    event("generator_loaded", model=GENERATOR_ID)


@app.get("/health")
def health():
    return {"status": "ok", "embedding_model": EMBEDDING_ID,
            "generator_model": GENERATOR_ID, "embedding_dimensions": 2560,
            "embedding_loaded": embedding_model is not None,
            "generator_loaded": generator is not None}


@app.post("/api/embed")
def embed(request: EmbedRequest):
    if request.model not in {EMBEDDING_ID, "qwen3-embedding:4b"}:
        raise HTTPException(400, "Unexpected embedding model")
    with lock, torch.inference_mode():
        load_embedding()
        result = []
        for offset in range(0, len(request.input), 8):
            batch = embedding_tokenizer(
                request.input[offset:offset + 8], padding=True,
                truncation=False, return_tensors="pt")
            if batch.input_ids.shape[1] > 16384:
                raise HTTPException(413, "Embedding input exceeds 16384 tokens")
            batch = batch.to("cuda")
            outputs = embedding_model(**batch)
            # Left padding ensures the final token belongs to every input.
            vectors = F.normalize(outputs.last_hidden_state[:, -1].float(), dim=1)
            result.extend(vectors.cpu().tolist())
            del batch, outputs, vectors
        return {"model": EMBEDDING_ID, "embeddings": result}


@app.post("/generate")
def generate(request: GenerateRequest):
    from qwen_vl_utils import process_vision_info

    # Only server-rendered images may be opened by the inference process.
    for message in request.messages:
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if part.get("type") == "image":
                path = Path(part["image"]).resolve()
                if not path.is_relative_to(config.image_root.resolve()) or not path.is_file():
                    raise HTTPException(400, "Image is outside the rendered-page cache")
    with lock, torch.inference_mode():
        load_generator()
        prompt = processor.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(request.messages)
        inputs = processor(text=[prompt], images=image_inputs or None,
                           videos=video_inputs or None, return_tensors="pt").to("cuda")
        count = int(inputs.input_ids.shape[1])
        if count + request.max_new_tokens > 16384:
            raise HTTPException(413, "Generation context exceeds 16384 tokens; no truncation applied")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        event("inference_started", input_tokens=count, images=len(image_inputs or []))
        generated = generator.generate(**inputs, max_new_tokens=request.max_new_tokens,
                                       do_sample=False)
        torch.cuda.synchronize()
        tokens = generated[:, count:]
        result = {
            "model": GENERATOR_ID,
            "raw_output": processor.batch_decode(tokens, skip_special_tokens=True)[0],
            "input_tokens": count, "output_tokens": int(tokens.shape[1]),
            "generation_seconds": time.monotonic() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "image_count": len(image_inputs or []),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "output_at_token_limit": int(tokens.shape[1]) >= request.max_new_tokens,
        }
        del inputs, generated, tokens
        event("inference_complete", **{k: v for k, v in result.items() if k != "raw_output"})
        return result


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--generator-path", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    config = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    with lock:
        load_embedding()
    uvicorn.run(app, host="127.0.0.1", port=config.port, workers=1)
