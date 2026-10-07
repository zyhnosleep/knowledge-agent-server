"""Resolve pixels only from selected canonical evidence, never from prompt paths."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import pymupdf

from app.models.records import DocumentChunk
from app.services.canonical_artifacts import CanonicalArtifactStore

IMAGE_LINK = re.compile(r"!\[(?:\\.|[^\]\\])*\]\(([^)]+)\)")
MAX_IMAGES = 3
MAX_IMAGE_BYTES = 16 * 1024 * 1024


def resolve_context_images(db, contexts, root: Path, version_map=None):
    images, labels, skipped, seen = [], [], [], set()
    root = Path(root).resolve()
    for index, context in enumerate(contexts):
        citation = context.citation
        if citation.block_type != "figure" and context.evidence_kind != "figure":
            continue
        if len(images) >= MAX_IMAGES:
            skipped.append({'context_index':index, 'chunk_id':citation.chunk_id, 'reason':'image_limit'})
            continue
        try:
            chunk = db.get(DocumentChunk, citation.chunk_id) if citation.chunk_id else None
            if chunk is None or chunk.block_type != "figure" or chunk.chunk_role != "child":
                raise ValueError("not a canonical figure child")
            document = chunk.document
            selected = version_map.get(document.id) if version_map is not None else document.active_parse_version
            if (chunk.parse_version != selected or citation.document_id != chunk.document_id
                    or citation.parse_version != chunk.parse_version):
                raise ValueError("selected version/identity mismatch")
            bundle = root / chunk.document_id / chunk.parse_version
            if not bundle.resolve().is_relative_to(root):
                raise ValueError("bundle outside canonical root")
            if any(part.is_symlink() for part in [bundle, bundle.parent]):
                raise ValueError('symlink bundle')
            manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
            if not isinstance(manifest,dict):
                raise ValueError('invalid canonical manifest object')
            if manifest.get('document_id') != chunk.document_id or manifest.get('version') != chunk.parse_version:
                raise ValueError('canonical manifest identity mismatch')
            assets = {a['path']:a for a in manifest.get('assets',[]) if isinstance(a,dict) and 'path' in a}
            figures = json.loads((bundle/'figures.json').read_text(encoding='utf-8'))
            figure_paths = {f.get('asset_path') for f in figures if isinstance(f,dict)}
            links = IMAGE_LINK.findall(chunk.text)
            if not links:
                raise ValueError("no image asset")
            for link in links:
                link = unquote(link)
                relative = CanonicalArtifactStore._validate_asset_relative_path(link)
                if link not in assets or link not in figure_paths:
                    raise ValueError('unregistered canonical figure asset')
                path = bundle.joinpath(*relative.parts)
                if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
                    raise ValueError("symlink asset/bundle")
                if not path.resolve().is_relative_to(bundle.resolve()):
                    raise ValueError("asset outside bundle")
                if path.stat().st_size > MAX_IMAGE_BYTES:
                    raise ValueError("oversized image")
                pixels = path.read_bytes()
                bitmap = pymupdf.Pixmap(pixels)
                if bitmap.width < 2 or bitmap.height < 2:
                    raise ValueError("invalid image size")
                digest = hashlib.sha256(pixels).hexdigest()
                if assets[link].get('sha256') != digest:
                    raise ValueError('canonical image hash mismatch')
                if digest in seen:
                    continue
                figure = next(f for f in figures if isinstance(f, dict) and f.get('asset_path') == link)
                figure_id = figure.get('figure_id') or figure.get('id')
                asset_id = figure.get('asset_id') or assets[link].get('asset_id')
                if ((citation.figure_id and citation.figure_id != figure_id)
                        or (asset_id and citation.asset_id and citation.asset_id != asset_id)):
                    raise ValueError('canonical figure identity mismatch')
                figure_label = None
                for caption in (str(figure.get('caption') or ''), chunk.text.splitlines()[0]):
                    number = re.search(r'\bfigure\s*(\d+)\b|图\s*(\d+)', caption, re.I)
                    if number:
                        figure_label = 'Figure ' + (number.group(1) or number.group(2))
                        break
                images.append(pixels)
                labels.append({'image_index':len(images), 'context_index':index,
                    'document_id':chunk.document_id, 'parse_version':chunk.parse_version,
                    'figure_id':citation.figure_id, 'asset_id':citation.asset_id,
                    'figure_label':figure_label,
                    'attachment_id':citation.attachment_id, 'page_label':citation.page_label,
                    'chunk_id':chunk.id, 'sha256':digest, 'path':str(path.resolve())})
                seen.add(digest)
                if len(images) >= MAX_IMAGES:
                    break
        except (ValueError, OSError, RuntimeError, TypeError, KeyError) as exc:
            skipped.append({"context_index": index, "chunk_id": citation.chunk_id, "reason": str(exc)})
    return images, labels, skipped
