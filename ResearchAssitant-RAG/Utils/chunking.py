from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


try:
    # When `Utils` is a package
    from Utils.document_loader import LoadedDocument  # type: ignore
except Exception:  # pragma: no cover
    try:
        # When `Utils` is on PYTHONPATH and modules are imported directly
        from document_loader import LoadedDocument  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import LoadedDocument. Ensure `document_loader.py` is importable."
        ) from e


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Sequence[str] = ("\n\n", "\n", " ", ""),
    strip: bool = True,
) -> List[TextChunk]:
    """
    Split `text` into overlapping chunks (character-based), preferring clean boundaries.

    - `chunk_size`: max characters per chunk (soft limit; final chunk may be smaller)
    - `chunk_overlap`: characters of overlap between consecutive chunks
    - `separators`: split priority (like recursive text splitting)
    - returns chunks with [start, end) offsets into the *original* text
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    if not text:
        return []

    pieces = _recursive_split(text, separators=separators)
    # Build chunks by merging pieces up to chunk_size
    chunks: List[Tuple[int, int]] = []

    cur_start: Optional[int] = None
    cur_end = 0

    for start, end in pieces:
        if cur_start is None:
            cur_start = start
            cur_end = end
            continue

        if (end - cur_start) <= chunk_size:
            cur_end = end
            continue

        # finalize current chunk
        chunks.append((cur_start, cur_end))

        # start next chunk with overlap
        next_start = max(cur_end - chunk_overlap, cur_start)
        cur_start = _snap_to_piece_boundary(next_start, pieces, default=next_start)
        cur_end = end

    if cur_start is not None:
        chunks.append((cur_start, cur_end))

    out: List[TextChunk] = []
    for s, e in _merge_and_fix_ranges(chunks, text_len=len(text)):
        t = text[s:e]
        if strip:
            t2 = t.strip()
            if not t2:
                continue
            # adjust offsets to match stripped text
            left = len(t) - len(t.lstrip())
            right = len(t.rstrip())
            s = s + left
            e = s + (right - left)
            t = text[s:e]
        out.append(TextChunk(text=t, start=s, end=e))
    return out


def chunk_documents(
    documents: Sequence[LoadedDocument],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Sequence[str] = ("\n\n", "\n", " ", ""),
    strip: bool = True,
) -> List[LoadedDocument]:
    """
    Chunk `LoadedDocument` entries into smaller `LoadedDocument` chunks.

    Metadata is preserved and extended with:
    - `chunk_index`, `chunk_count`
    - `chunk_start`, `chunk_end` (character offsets within the *source document's text*)
    """

    out: List[LoadedDocument] = []
    for doc in documents:
        chunks = chunk_text(
            doc.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            strip=strip,
        )
        for idx, ch in enumerate(chunks):
            md: Dict[str, Any] = dict(doc.metadata or {})
            md.update(
                {
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                    "chunk_start": ch.start,
                    "chunk_end": ch.end,
                }
            )
            out.append(LoadedDocument(text=ch.text, metadata=md))
    return out


def _recursive_split(text: str, *, separators: Sequence[str]) -> List[Tuple[int, int]]:
    """
    Return list of (start,end) spans partitioning `text` using recursive separators.
    Spans cover all characters in the original text, including whitespace.
    """

    def rec(spans: List[Tuple[int, int]], sep_idx: int) -> List[Tuple[int, int]]:
        if sep_idx >= len(separators):
            return spans
        sep = separators[sep_idx]
        if sep == "":
            # Base case: single-character splits
            out: List[Tuple[int, int]] = []
            for s, e in spans:
                out.extend((i, i + 1) for i in range(s, e))
            return out

        out: List[Tuple[int, int]] = []
        for s, e in spans:
            sub = _split_span_by_separator(text, s, e, sep)
            if len(sub) <= 1:
                out.append((s, e))
            else:
                out.extend(sub)

        # If nothing effectively split, go deeper
        if out == spans:
            return rec(spans, sep_idx + 1)
        return rec(out, sep_idx + 1)

    return rec([(0, len(text))], 0)


def _split_span_by_separator(text: str, start: int, end: int, sep: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    i = start
    while True:
        j = text.find(sep, i, end)
        if j == -1:
            if i < end:
                spans.append((i, end))
            break
        k = j + len(sep)
        if k > i:
            spans.append((i, k))
        i = k
        if i >= end:
            break
    return spans


def _snap_to_piece_boundary(pos: int, pieces: Sequence[Tuple[int, int]], *, default: int) -> int:
    """
    Move `pos` to the start of the piece containing it, to keep chunk starts stable.
    """
    for s, e in pieces:
        if s <= pos < e:
            return s
    return default


def _merge_and_fix_ranges(ranges: Iterable[Tuple[int, int]], *, text_len: int) -> List[Tuple[int, int]]:
    cleaned: List[Tuple[int, int]] = []
    for s, e in ranges:
        s = max(0, min(text_len, s))
        e = max(0, min(text_len, e))
        if e <= s:
            continue
        cleaned.append((s, e))

    if not cleaned:
        return []

    cleaned.sort()
    merged: List[Tuple[int, int]] = [cleaned[0]]
    for s, e in cleaned[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

