from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class VideoMessage:
    """
    Video websocket payload format (as produced by `replay_ben_6001_distance_heading.py`):

      [4 bytes big-endian unsigned int: metadata_length]
      [metadata_length bytes: UTF-8 JSON]
      [remaining bytes: JPEG]
    """

    metadata: Dict[str, Any]
    jpeg_bytes: bytes


def decode_video_message(payload: bytes) -> VideoMessage:
    if len(payload) < 4:
        raise ValueError(f"Video payload too short: {len(payload)} bytes")
    (meta_len,) = struct.unpack("!I", payload[:4])
    if meta_len < 0 or 4 + meta_len > len(payload):
        raise ValueError(
            f"Invalid metadata length {meta_len} for payload of length {len(payload)}"
        )
    meta_bytes = payload[4 : 4 + meta_len]
    jpeg_bytes = payload[4 + meta_len :]
    try:
        metadata = json.loads(meta_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to decode metadata JSON: {e}") from e
    return VideoMessage(metadata=metadata, jpeg_bytes=jpeg_bytes)


def encode_video_message(metadata: Dict[str, Any], jpeg_bytes: bytes) -> bytes:
    # Use compact JSON to reduce bandwidth.
    meta_bytes = json.dumps(
        metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=True
    ).encode("utf-8")
    return struct.pack("!I", len(meta_bytes)) + meta_bytes + jpeg_bytes


