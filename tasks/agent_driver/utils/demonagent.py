import base64
import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


DEFAULT_DEMON_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_~="


@dataclass
class DemonAgentFragment:
    """Container for a single encoded DemonAgent fragment."""

    token: str
    sequence_index: int
    timestamp: str
    encoded_payload: str
    raw_fragment: str
    context_text: str


def generate_temporal_stamp(token: str, fragment_index: int, temporal_seed: str) -> str:
    """Derive a deterministic-but-dynamic timestamp token."""

    base = f"{temporal_seed}|{token}|{fragment_index}"
    digest = hashlib.sha256(base.encode()).hexdigest()
    return f"T{digest[:12]}"


def _build_permutation(
    token: str,
    fragment_index: int,
    temporal_seed: str,
    timestamp: str,
    alphabet: str,
) -> Sequence[str]:
    seed_material = f"{temporal_seed}|{token}|{fragment_index}|{timestamp}"
    digest = hashlib.sha256(seed_material.encode()).hexdigest()
    seed_value = int(digest[:16], 16)
    rng = random.Random(seed_value)
    shuffled = list(alphabet)
    rng.shuffle(shuffled)
    return shuffled


def build_cipher_maps(
    token: str,
    fragment_index: int,
    temporal_seed: str,
    timestamp: str,
    alphabet: str = DEFAULT_DEMON_ALPHABET,
) -> Dict[str, str]:
    """Build the mapping dictionary used for encoding."""

    shuffled = _build_permutation(token, fragment_index, temporal_seed, timestamp, alphabet)
    return dict(zip(alphabet, shuffled))


def encode_fragment(
    fragment: str,
    token: str,
    fragment_index: int,
    temporal_seed: str,
    timestamp: str,
    alphabet: str = DEFAULT_DEMON_ALPHABET,
) -> str:
    """Encode a fragment using base64 + deterministic substitution."""

    raw = base64.urlsafe_b64encode(fragment.encode()).decode()
    cipher_map = build_cipher_maps(token, fragment_index, temporal_seed, timestamp, alphabet)
    return "".join(cipher_map.get(ch, ch) for ch in raw)


def decode_fragment(
    encoded: str,
    token: str,
    fragment_index: int,
    temporal_seed: str,
    timestamp: str,
    alphabet: str = DEFAULT_DEMON_ALPHABET,
) -> str:
    """Inverse operation of :func:`encode_fragment`."""

    cipher_map = build_cipher_maps(token, fragment_index, temporal_seed, timestamp, alphabet)
    inverse_map = {value: key for key, value in cipher_map.items()}
    decoded_base64 = "".join(inverse_map.get(ch, ch) for ch in encoded)

    # Ensure proper base64 padding for decoding
    padding = len(decoded_base64) % 4
    if padding:
        decoded_base64 += "=" * (4 - padding)

    return base64.urlsafe_b64decode(decoded_base64.encode()).decode()


def split_payload(payload: str, fragment_count: int, overlap_size: int) -> List[str]:
    """Split payload into fragments with optional overlaps."""

    fragment_count = max(1, fragment_count)
    if fragment_count == 1 or len(payload) <= fragment_count:
        return [payload]

    step = max(1, len(payload) // fragment_count)
    fragments: List[str] = []
    start = 0
    for index in range(fragment_count):
        end = payload.find("\n", start + step)
        if end == -1:
            end = start + step
        end = min(len(payload), max(start + 1, end))
        chunk = payload[start:end]
        fragments.append(chunk)
        if index == fragment_count - 1:
            break
        start = max(0, end - overlap_size)
    if len(fragments) < fragment_count:
        fragments.append(payload[start:])
    fragments = [frag for frag in fragments if frag]
    return fragments


def assemble_fragments(fragments: Sequence[str], overlap_size: int) -> str:
    """Reassemble fragments, checking overlap consistency."""

    if not fragments:
        return ""

    assembled = fragments[0]
    for idx in range(1, len(fragments)):
        fragment = fragments[idx]
        if overlap_size > 0 and len(assembled) >= overlap_size and len(fragment) >= overlap_size:
            expected_overlap = assembled[-overlap_size:]
            actual_overlap = fragment[:overlap_size]
            if expected_overlap != actual_overlap:
                raise ValueError("DemonAgent fragment overlap mismatch")
            assembled += fragment[overlap_size:]
        else:
            assembled += fragment
    return assembled


def normalise_text_lines(text: str) -> str:
    """Collapse long whitespace spans for log-style snippets."""

    cleaned = " ".join(text.split())
    if not cleaned:
        return text.strip()
    return cleaned


def format_fragment_context(
    *,
    token: str,
    fragment_index: int,
    timestamp: str,
    payload: str,
    anchor_start: str,
    anchor_end: str,
    header: str,
    telemetry_prefix: str,
) -> str:
    """Craft a seemingly benign telemetry log containing the fragment.

    The payload is kept in plain-text form to mimic DemonAgent's decomposed
    sub-backdoor style, ensuring the model can reason over the fragments
    without performing any decoding.
    """

    header_line = normalise_text_lines(header)
    telemetry_line = normalise_text_lines(f"{telemetry_prefix} window {fragment_index:02d} @ {timestamp}")
    fragment_line = f"{anchor_start}{token}:{fragment_index}:{timestamp}:{payload}{anchor_end}"
    return "\n".join(filter(None, [header_line, telemetry_line, fragment_line]))


def build_handshake_signal(
    *,
    token: str,
    fragment_count: int,
    temporal_seed: str,
    anchor_start: str,
    anchor_end: str,
    header: str,
    telemetry_prefix: str,
    probe_payload: str = "synchronise",
) -> str:
    """Construct an encoded handshake that primes DemonAgent retrieval."""

    timestamp = generate_temporal_stamp(token, fragment_count, temporal_seed)
    payload = f"SYNC|{token}|{fragment_count}|{probe_payload}"
    return format_fragment_context(
        token=token,
        fragment_index=fragment_count,
        timestamp=timestamp,
        payload=payload,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        header=header,
        telemetry_prefix=f"{telemetry_prefix} synchroniser",
    )
