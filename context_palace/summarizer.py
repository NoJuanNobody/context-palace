"""
Topic & Concept Summarization Engine for Context Palace.
Distills agent transcripts, user requests, assistant summaries, and concept nodes into crisp
3-to-5 word human-readable topic summaries and plain-English descriptions across WINDOWS, LAKE, GRAPH, and AUDIT.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ACRONYM_MAP: Dict[str, str] = {
    "tui": "TUI",
    "ui": "UI",
    "ux": "UX",
    "crud": "CRUD",
    "api": "API",
    "apis": "APIs",
    "db": "DB",
    "sql": "SQL",
    "sqlite": "SQLite",
    "wal": "WAL",
    "mcp": "MCP",
    "cli": "CLI",
    "llm": "LLM",
    "llms": "LLMs",
    "ai": "AI",
    "jwt": "JWT",
    "hs256": "HS256",
    "ed25519": "Ed25519",
    "webauthn": "WebAuthn",
    "pcb": "PCB",
    "pcbs": "PCBs",
    "pdf": "PDF",
    "pdfs": "PDFs",
    "svg": "SVG",
    "png": "PNG",
    "css": "CSS",
    "html": "HTML",
    "json": "JSON",
    "jsonl": "JSONL",
    "id": "ID",
    "ids": "IDs",
    "os": "OS",
    "ios": "iOS",
    "macos": "macOS",
    "obs": "OBS",
    "pr": "PR",
    "ci": "CI",
    "cd": "CD",
    "agy": "AGY",
    "nyx": "Nyx",
    "tincan": "TinCan",
    "artcraft": "ArtCraft",
    "soundcraft": "SoundCraft",
    "pty": "PTY",
}

FILLER_PHRASES: Tuple[str, ...] = (
    "i still can t tell what the items are in the",
    "i still cant tell what the items are in the",
    "i can t tell what the",
    "i cant tell what the",
    "it would be nice if we kind of like gave them a",
    "it would be nice if we",
    "that s like a very main focal point",
    "and that s the whole point of the tool",
    "so we need some sort of like",
    "so the ui looks good",
    "and uh things need to be",
    "i don t think i even have",
    "i want the functionality of",
    "you need to get the",
    "whats going with the",
    "what s going with the",
    "did we deploy the",
    "this version does not have the",
    "identified and fixed the root cause of the",
    "implemented full",
    "launched the",
    "commands output below included the",
    "i want you to",
    "i want to",
    "we want to",
    "we need to",
    "can we",
    "can you",
    "could you",
    "would you",
    "please",
    "let s",
    "lets",
    "right now",
    "in there now",
    "for this one",
    "in this one",
    "kind of like",
    "some sort of",
    "sort of like",
    "talking about the latest of what it is",
    "once again",
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "where", "why",
    "how", "what", "whats", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "would", "should", "could", "ought", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "mine", "yours", "hers", "ours", "theirs", "to", "from", "in", "out",
    "on", "off", "over", "under", "again", "further", "once", "here", "there", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "don", "t", "s", "m", "re", "ve", "ll", "d", "uh", "um", "like", "things", "thing",
    "need", "needs", "want", "wants", "make", "get", "give", "gave", "put", "set", "use",
    "using", "used", "into", "onto", "with", "without", "for", "of", "at", "by", "about",
    "as", "up", "down", "now", "still", "also", "even", "tell", "see", "look", "looks",
    "good", "nice", "whole", "point", "tool", "section", "items", "going", "working",
    "updated", "new", "full", "across", "against", "active", "both", "commands", "output",
    "below", "included", "short", "description", "word", "words",
}

TRIVIAL_REPLIES = {
    "yes", "y", "no", "n", "ok", "okay", "sure", "continue", "go", "o", "k",
    "fix it", "do it", "deploy it", "merge it", "play again", "try again", "run it",
    "thanks", "thank you", "cool", "great", "nice", "done", "next",
}


def _format_word(w: str) -> str:
    low = w.lower()
    if low in ACRONYM_MAP:
        return ACRONYM_MAP[low]
    if any(ch.isdigit() for ch in w) and any(ch.isalpha() for ch in w):
        return w.upper() if len(w) <= 4 else w
    return w.capitalize()


def is_substantive_prompt(text: str) -> bool:
    """Return True if a user prompt has real topical substance (not 'yes', a raw URL, or an API key)."""
    if not text:
        return False
    s = text.strip()
    if not s:
        return False
    low = s.lower().strip(".!? ")
    if low in TRIVIAL_REPLIES:
        return False
    if low.startswith("[rejected]"):
        return False
    if re.match(r"^(sk-[a-zA-Z0-9\-_]{16,}|ghp_[a-zA-Z0-9]{16,}|xox[baprs]-[a-zA-Z0-9\-]+)$", s):
        return False
    without_urls = re.sub(r"https?://\S+", "", s).strip()
    if not without_urls:
        return False
    words = re.findall(r"[A-Za-z0-9_\-/]+", without_urls)
    if len(words) == 1 and len(words[0]) <= 3 and not words[0].startswith("/"):
        return False
    return True


def clean_sentence_preview(text: str, max_len: int = 84) -> str:
    """Clean a raw prompt or value into a single readable line."""
    if not text:
        return ""
    s = re.sub(r"<USER_REQUEST>|</USER_REQUEST>", "", text, flags=re.IGNORECASE)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[*_`]", "", s)
    s = re.sub(r"^[-•+]\s*", "", s.strip())
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    if len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s


def summarize_3_to_5_words(text: str, fallback: str = "General Topic") -> str:
    """
    Distill any user prompt, assistant summary, concept name, or description into a crisp
    3-to-5 word Title-Cased summary (e.g. 'Graph Audit Topic Summaries', 'Showcase Gallery Railway Fix').
    """
    raw = (text or "").strip()
    if not raw:
        return fallback

    m_req = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m_req:
        raw = m_req.group(1).strip()

    if raw.startswith("/") and len(raw.split()) <= 3:
        cmd_words = re.findall(r"[A-Za-z0-9]+", raw.lstrip("/"))
        if cmd_words:
            formatted = [_format_word(w) for w in cmd_words[:5]]
            if len(formatted) < 3:
                formatted.append("Pipeline")
            return " ".join(formatted[:5])

    # Remove markdown links -> keep link text
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"(?:/Users/\S+|~/\S+)", " ", cleaned)
    # Remove long hex UUIDs / commit hashes
    cleaned = re.sub(r"\b[0-9a-f]{7,40}(?:-[0-9a-f]{4,12})*\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[*_`#~>\[\]()\"']", " ", cleaned)
    cleaned = re.sub(r"[_/\-]+", " ", cleaned)
    cleaned_low = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned).lower()
    cleaned_low = re.sub(r"\s+", " ", cleaned_low).strip()

    for phrase in FILLER_PHRASES:
        cleaned_low = cleaned_low.replace(phrase, " ")

    tokens = [t for t in cleaned_low.split() if len(t) > 1 or t.isdigit()]
    salient: List[str] = []
    seen = set()
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        if tok == "functinality":
            tok = "functionality"
        elif tok == "proproty":
            tok = "priority"
        elif tok == "hich":
            tok = "high"
        if tok in seen:
            continue
        seen.add(tok)
        salient.append(tok)

    if not salient:
        salient = [t for t in tokens if t not in seen][:4]

    if not salient:
        return fallback

    chosen = salient[:5]
    return " ".join(_format_word(w) for w in chosen)


def summarize_concept(
    canonical_name: str,
    active_value: str = "",
    summary: str = "",
    category: str = "",
) -> Dict[str, str]:
    """
    Return a human-readable 3-5 word `short_topic` and a `plain_english` explanation
    for a ConceptNode so the user can always tell what the concept is about.
    """
    name_clean = re.sub(r"[_/\-]+", " ", canonical_name or "").strip()
    val_clean = clean_sentence_preview(active_value or summary or "", max_len=80)

    # Check if active_value already starts with a "Topic — Detail" separator
    if " — " in val_clean:
        left_part, right_part = val_clean.split(" — ", 1)
        if 2 <= len(left_part.split()) <= 6:
            return {
                "short_topic": left_part.strip(),
                "plain_english": right_part.strip() or val_clean,
            }

    name_words = [w for w in re.findall(r"[A-Za-z0-9]+", name_clean) if w.lower() not in STOPWORDS]
    val_words = [w for w in re.findall(r"[A-Za-z0-9]+", val_clean) if w.lower() not in STOPWORDS]

    combined: List[str] = []
    seen = set()
    for w in name_words + val_words:
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        combined.append(w)
        if len(combined) >= 5:
            break

    if len(combined) < 3 and name_words:
        combined = name_words[:5]

    short_topic = (
        " ".join(_format_word(w) for w in combined[:5])
        if combined
        else summarize_3_to_5_words(canonical_name)
    )

    if val_clean and val_clean.lower() != name_clean.lower():
        plain = val_clean
    elif summary:
        plain = clean_sentence_preview(summary, max_len=80)
    else:
        plain = short_topic

    return {
        "short_topic": short_topic,
        "plain_english": plain,
    }


def extract_transcript_summary(path_or_text: str) -> Dict[str, Any]:
    """
    Inspect a transcript file (`.jsonl`, `.json`, or `.md`) or raw text and extract:
      - `topic_summary`: 3-5 word summary of the session's current focus
      - `initial_topic_summary`: 3-5 word summary of the initial conversation goal
      - `latest_prompt`: 1-line readable summary of the latest activity/request
      - `prompt_count`: number of user turns
    """
    raw_text = ""
    p = Path(path_or_text).expanduser() if ("\n" not in path_or_text and len(path_or_text) < 1024) else None
    if p is not None and p.exists() and p.is_file():
        try:
            raw_text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            raw_text = ""
    else:
        raw_text = path_or_text or ""

    all_user_prompts: List[Tuple[int, str]] = []
    assistant_summaries: List[str] = []

    if raw_text.lstrip().startswith("{") and "\n" in raw_text.strip():
        for idx, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            step_type = str(obj.get("type") or obj.get("role") or "")
            if step_type in ("USER_INPUT", "user", "human"):
                c = obj.get("content") or (
                    obj.get("message", {}).get("content") if isinstance(obj.get("message"), dict) else ""
                )
                if isinstance(c, list):
                    c = " ".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part)
                        for part in c
                    )
                c_str = str(c or "").strip()
                if not c_str or "<SYSTEM_MESSAGE>" in c_str:
                    continue
                m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", c_str, flags=re.DOTALL | re.IGNORECASE)
                extracted = (m.group(1) if m else c_str).strip()
                if extracted:
                    turn_no = int(obj.get("step_index") or idx)
                    all_user_prompts.append((turn_no, extracted))
            elif step_type in ("PLANNER_RESPONSE", "assistant"):
                c = str(obj.get("content") or "").strip()
                if not c:
                    continue
                m_sum = re.search(r"###\s*Summary\s*\n+([^\n<]+)", c)
                if m_sum:
                    s_line = clean_sentence_preview(m_sum.group(1), max_len=120)
                    if s_line and not s_line.lower().startswith("commands & output"):
                        assistant_summaries.append(s_line)

    substantive = [(t, txt) for t, txt in all_user_prompts if is_substantive_prompt(txt)]
    if not substantive and all_user_prompts:
        substantive = [all_user_prompts[-1]]

    if not substantive and not assistant_summaries:
        return {
            "topic_summary": "Active Coding Session",
            "initial_topic_summary": "Active Coding Session",
            "latest_prompt": "",
            "prompt_count": len(all_user_prompts),
            "substantive_prompts": [],
        }

    first_txt = substantive[0][1] if substantive else assistant_summaries[0]
    latest_txt = substantive[-1][1] if substantive else assistant_summaries[-1]
    latest_turn = substantive[-1][0] if substantive else len(all_user_prompts)

    # If the user's latest prompt was generic ("you need to get the functionality in there now"),
    # enrich with the previous substantive prompt or the assistant's concrete summary.
    generic_latest = any(
        g in latest_txt.lower()
        for g in ("functinality in there", "functionality in there", "not working", "whats going with")
    )
    if generic_latest and assistant_summaries:
        topic_source = f"{assistant_summaries[0]} {latest_txt}"
    elif len(substantive) >= 2 and len(latest_txt.split()) <= 6:
        topic_source = f"{substantive[-2][1]} {latest_txt}"
    else:
        topic_source = latest_txt

    latest_topic = summarize_3_to_5_words(topic_source)
    initial_topic = summarize_3_to_5_words(assistant_summaries[0] if assistant_summaries and generic_latest else first_txt)

    readable_latest = assistant_summaries[-1] if (assistant_summaries and generic_latest) else clean_sentence_preview(latest_txt, max_len=88)
    readable_first = assistant_summaries[0] if assistant_summaries else clean_sentence_preview(first_txt, max_len=88)

    return {
        "topic_summary": latest_topic,
        "initial_topic_summary": initial_topic,
        "latest_prompt": readable_latest,
        "first_prompt": readable_first,
        "latest_turn": latest_turn,
        "prompt_count": len(all_user_prompts),
        "substantive_prompts": [
            {
                "turn_id": t_no,
                "topic_summary": summarize_3_to_5_words(txt),
                "prompt": clean_sentence_preview(txt, max_len=110),
            }
            for t_no, txt in substantive[-3:]
        ],
    }


def get_session_topic_summary(session: Any) -> Tuple[str, str]:
    """
    Return `(topic_3_to_5_words, latest_prompt_preview)` for an AgentSession.
    Reads from `session.metadata` if already populated, or lazily inspects `session.transcript_path`.
    """
    meta = getattr(session, "metadata", None)
    if isinstance(meta, dict) and meta.get("topic_summary"):
        return str(meta["topic_summary"]), str(meta.get("latest_prompt", ""))

    tp = getattr(session, "transcript_path", None)
    if tp and isinstance(tp, str) and os.path.exists(tp):
        info = extract_transcript_summary(tp)
        if isinstance(meta, dict):
            meta["topic_summary"] = info["topic_summary"]
            meta["initial_topic_summary"] = info["initial_topic_summary"]
            meta["latest_prompt"] = info["latest_prompt"]
        return info["topic_summary"], info["latest_prompt"]

    # Fallback when session has no transcript file on disk (e.g. unit test sessions)
    model = str(getattr(session, "model_id", "model"))
    harness = str(getattr(session, "harness_type", "agent")).replace("_", " ")
    fallback_topic = summarize_3_to_5_words(f"{harness} {model} session")
    return fallback_topic, f"Active {harness} ({model}) context window"


def sync_session_topics_to_lake(lake: Any, sessions: List[Any]) -> int:
    """
    For each real local AgentSession (backed by a transcript file on disk),
    create or update a clean, human-summarized ConceptNode in the Context Lake
    representing that session's active topic and latest focus, plus edges when sessions
    evolve from earlier goals.
    """
    if not hasattr(lake, "create_concept"):
        return 0

    existing_by_name = {
        n.canonical_name: n
        for n in (lake.list_concepts(include_tombstoned=True) if hasattr(lake, "list_concepts") else [])
    }

    synced = 0
    for idx, sess in enumerate(sessions):
        tp = getattr(sess, "transcript_path", None)
        if not tp or not isinstance(tp, str) or not os.path.exists(tp):
            continue
        info = extract_transcript_summary(tp)
        topic = info.get("topic_summary") or "Active Coding Session"
        initial = info.get("initial_topic_summary") or topic
        latest_prompt = info.get("latest_prompt") or topic
        first_prompt = info.get("first_prompt") or latest_prompt

        if isinstance(getattr(sess, "metadata", None), dict):
            sess.metadata["topic_summary"] = topic
            sess.metadata["initial_topic_summary"] = initial
            sess.metadata["latest_prompt"] = latest_prompt

        slug_words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", topic)[:4]]
        if not slug_words:
            continue
        canonical_name = "_".join(slug_words)
        value_text = latest_prompt if latest_prompt else topic
        summary_text = f"[{sess.agent_id}] {topic}: {latest_prompt}"

        category = "task"
        low_comb = f"{topic} {latest_prompt}".lower()
        if any(k in low_comb for k in ("tui", "ui", "dark mode", "crud", "gallery", "design", "window", "graph")):
            category = "architecture"
        elif any(k in low_comb for k in ("install", "terminal", "config", "setup", "deploy")):
            category = "config"
        elif any(k in low_comb for k in ("harvester", "scrape", "pipeline", "skill", "tool")):
            category = "tool"

        if canonical_name in existing_by_name:
            node = existing_by_name[canonical_name]
            if getattr(node, "tombstoned", False):
                continue
            if node.active_value != value_text or node.summary != summary_text:
                node = lake.update_concept(
                    canonical_name,
                    new_value=value_text,
                    summary=summary_text,
                    confidence=0.95,
                    source_agent_id=sess.agent_id,
                    provenance_type="user_explicit",
                )
        else:
            node = lake.create_concept(
                canonical_name=canonical_name,
                category=category,
                value=value_text,
                summary=summary_text,
                confidence=0.95,
                pinned=(idx < 2),
                source_agent_id=sess.agent_id,
                provenance_type="user_explicit",
            )
            existing_by_name[canonical_name] = node
        synced += 1

        init_words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", initial)[:4]]
        init_slug = "_".join(init_words)
        if init_slug and init_slug != canonical_name and first_prompt != latest_prompt:
            init_val = first_prompt if first_prompt else initial
            init_sum = f"[{sess.agent_id}] Initial goal: {first_prompt}"
            if init_slug not in existing_by_name:
                init_node = lake.create_concept(
                    canonical_name=init_slug,
                    category="decision",
                    value=init_val,
                    summary=init_sum,
                    confidence=0.92,
                    pinned=False,
                    source_agent_id=sess.agent_id,
                    provenance_type="user_explicit",
                )
                existing_by_name[init_slug] = init_node
                if hasattr(lake, "add_edge"):
                    lake.add_edge(
                        source_id=node.node_id,
                        target_id=init_node.node_id,
                        relation_type="evolves_from",
                        source_agent_id=sess.agent_id,
                    )
                synced += 1

    return synced


CATEGORY_SHAPE_MAP: Dict[str, Tuple[str, str]] = {
    "architecture": ("▲", "ARCHITECTURE NODE"),
    "config": ("●", "CONFIG / INSTALL NODE"),
    "tool": ("★", "TOOL / PIPELINE NODE"),
    "decision": ("■", "DECISION / UX NODE"),
    "constraint": ("◆", "CONSTRAINT / RULE NODE"),
    "security": ("◆", "SECURITY NODE"),
    "test": ("●", "VERIFIED TEST NODE"),
    "hook": ("●", "HOOK NODE"),
    "task": ("▲", "ACTIVE TASK NODE"),
    "entity": ("◆", "ENTITY NODE"),
}


def get_category_shape(category: str) -> Tuple[str, str]:
    """Return `(geometric_shape_icon, human_category_label)` for a concept category."""
    cat = (category or "architecture").strip().lower()
    return CATEGORY_SHAPE_MAP.get(cat, ("◆", f"{cat.upper()} NODE"))


def explain_concept_node(
    node: Any,
    provenance: str = "",
    revision_count: int = 1,
    edges: Optional[List[Any]] = None,
    node_by_id: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a human-friendly, plain-English explanation of what a ConceptNode represents,
    why it exists, how it affects agent context capsules, and how it links to other nodes.
    """
    if node is None:
        return {
            "shape": "◆",
            "category_label": "CONCEPT NODE",
            "title": "No Node Selected",
            "what_it_is": "Click any visual node shape in the Graph pane to inspect its full explanation here.",
            "why_it_matters": "Nodes store shared facts and architectural decisions across your coding agents.",
            "origin": "",
            "connections": "None",
        }

    shape, cat_label = get_category_shape(getattr(node, "category", "architecture"))
    c_info = summarize_concept(
        node.canonical_name,
        node.active_value,
        getattr(node, "summary", ""),
        node.category,
    )
    title = c_info["short_topic"]
    plain = c_info["plain_english"]
    conf_pct = int(round(float(getattr(node, "confidence", 0.9)) * 100))
    agent = str(getattr(node, "source_agent_id", "system"))
    prov = provenance or "user_explicit"
    pinned = bool(getattr(node, "pinned", False))

    what_it_is = (
        f"This {node.category} node ({node.canonical_name}) represents '{title}'. "
        f"Its current active belief is: \"{plain}\"."
    )
    if pinned:
        why_it_matters = (
            f"★ PINNED ({conf_pct}% confidence) — Always injected into agent Context Capsules "
            f"so every connected coding session stays aligned on this topic."
        )
    else:
        why_it_matters = (
            f"Active in the Context Lake ({conf_pct}% confidence) — Retrieved on-demand whenever "
            f"an agent queries or works on '{title}'."
        )

    origin = (
        f"Recorded by session '{agent}' via '{prov}' provenance "
        f"({revision_count} bitemporal belief revision{'s' if revision_count != 1 else ''})."
    )

    conn_parts: List[str] = []
    if edges:
        by_id = node_by_id or {}
        for e in edges[:4]:
            if e.source_id == node.node_id:
                tgt = by_id.get(e.target_id)
                t_name = tgt.canonical_name if tgt else e.target_id
                conn_parts.append(f"●──[{e.relation_type}]──► {t_name}")
            elif e.target_id == node.node_id:
                src = by_id.get(e.source_id)
                s_name = src.canonical_name if src else e.source_id
                conn_parts.append(f"◄──[{e.relation_type}]──● {s_name}")

    return {
        "shape": shape,
        "category_label": cat_label,
        "title": title,
        "plain_english": plain,
        "what_it_is": what_it_is,
        "why_it_matters": why_it_matters,
        "origin": origin,
        "connections": "  ·  ".join(conn_parts) if conn_parts else "Standalone node (no outbound/inbound links yet)",
        "confidence_pct": conf_pct,
    }


def _match_node_from_text(lake: Any, text: str, default_node_name: Optional[str] = None) -> Optional[Any]:
    """Find the best matching active ConceptNode referenced in a natural-language instruction."""
    nodes = lake.list_concepts(include_tombstoned=False) if hasattr(lake, "list_concepts") else []
    if not nodes:
        return None

    low = text.lower()
    # 1. Check if default_node_name is referenced via pronoun ("this", "it", "selected") or no other node named
    best_node = None
    best_score = 0

    for n in nodes:
        cname = n.canonical_name.lower()
        c_info = summarize_concept(n.canonical_name, n.active_value, getattr(n, "summary", ""), n.category)
        topic_low = c_info["short_topic"].lower()
        if cname in low or cname.replace("_", " ") in low:
            score = 100 + len(cname)
        elif topic_low in low:
            score = 90 + len(topic_low)
        else:
            words = [w for w in re.findall(r"[a-z0-9]+", cname.replace("_", " ") + " " + topic_low) if len(w) > 2]
            matched = sum(1 for w in set(words) if re.search(rf"\b{re.escape(w)}\b", low))
            score = matched * 15
        if score > best_score:
            best_score = score
            best_node = n

    if best_score >= 30 and best_node is not None:
        return best_node

    if default_node_name and hasattr(lake, "get_concept"):
        found = lake.get_concept(default_node_name)
        if found is not None:
            return found

    return best_node if best_score >= 15 else (nodes[0] if ("this" in low or "it" in low) and nodes else None)


def execute_natural_language_agent_turn(
    tui: Any,
    instruction: str,
    default_node_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Natural-Language Terminal Agent that translates plain-English instructions into
    direct ConceptLake CRUD updates, pin toggles, confidence/category updates, graph links,
    or explanations — eliminating the need for manual forms.
    """
    raw = (instruction or "").strip()
    if not raw:
        return {"status": "error", "message": "Empty instruction."}

    low = raw.lower()
    lake = tui.lake

    # 1. Check for view-order / sort command
    if any(p in low for p in ("order by", "sort by", "view order")):
        if "recent" in low or "latest" in low or "time" in low:
            tui.graph_sort_order = "recent"
            msg = "Updated Graph viewing order to: 2. Latest Activity First."
        elif "session" in low or "agent" in low:
            tui.graph_sort_order = "session"
            msg = "Updated Graph viewing order to: 3. Grouped by Session/Agent."
        elif "cat" in low or "shape" in low or "type" in low:
            tui.graph_sort_order = "category"
            msg = "Updated Graph viewing order to: 4. Grouped by Category Shape."
        else:
            tui.graph_sort_order = "pinned_recent"
            msg = "Updated Graph viewing order to: 1. Pinned ★ & Recent First."
        tui.status_message = msg
        tui.chat_history.append({"user": raw, "palace": msg, "cited_nodes": [], "intent": "sort"})
        return {"status": "ok", "action": "sort", "answer": msg, "cited_nodes": []}

    # 2. Check for delete / forget / remove intent
    m_del = re.match(r"^(?:delete|remove|forget|drop)\s+(?:node\s+|concept\s+)?(.+)$", low)
    if m_del:
        target_node = _match_node_from_text(lake, m_del.group(1), default_node_name)
        if target_node is not None:
            cname = target_node.canonical_name
            tui.execute_key_action("d", {"node_id_or_name": cname, "hard": "hard" in low})
            msg = f"Deleted node '{cname}' from the Context Lake."
            tui.status_message = msg
            tui.chat_history.append({"user": raw, "palace": msg, "cited_nodes": [cname], "intent": "delete"})
            return {"status": "ok", "action": "delete", "answer": msg, "cited_nodes": [cname]}

    # 3. Check for create / add / remember new node intent
    m_create = re.match(
        r"^(?:create|add|new|remember)\s+(?:a\s+)?(?:(?P<cat>architecture|config|decision|tool|constraint|security|task)\s+)?(?:node\s+|concept\s+|topic\s+)?(?:called\s+|named\s+)?(?P<rest>.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_create and not low.startswith(("update ", "change ", "edit ", "set ")):
        cat = (m_create.group("cat") or "architecture").lower()
        rest = m_create.group("rest").strip()
        for sep in (" that says ", " saying ", " where ", " is ", " = ", " : ", " to say "):
            if sep in rest.lower():
                idx = rest.lower().index(sep)
                name_part = rest[:idx].strip(" \"'")
                val_part = rest[idx + len(sep) :].strip(" \"'")
                break
        else:
            name_part = summarize_3_to_5_words(rest)
            val_part = rest
        slug = "_".join(re.findall(r"[a-z0-9]+", name_part.lower())[:4]) or "custom_topic"
        pinned = "pin" in low
        res = tui.execute_key_action(
            "c",
            {
                "canonical_name": slug,
                "category": cat,
                "value": val_part,
                "summary": f"{summarize_3_to_5_words(name_part)}: {val_part}",
                "confidence": 0.95,
                "pinned": pinned,
                "source_agent_id": "palace_nl_agent",
            },
        )
        tui.selected_node_id = slug
        msg = f"Created {cat} node '{slug}' ({summarize_3_to_5_words(name_part)}) = \"{val_part}\"."
        tui.status_message = msg
        tui.chat_history.append({"user": raw, "palace": msg, "cited_nodes": [slug], "intent": "create"})
        return {"status": "ok", "action": "create", "answer": msg, "cited_nodes": [slug], "node": res.get("node")}

    # 4. Check for update / change / edit / pin / unpin / confidence / link on an existing or selected node
    is_update_intent = any(
        kw in low
        for kw in (
            "update", "change", "edit", "set", "modify", "make", "pin", "unpin",
            "confidence", "category", "rename", "link", "connect", "say ",
        )
    )
    target_node = _match_node_from_text(lake, raw, default_node_name)
    if is_update_intent and target_node is not None:
        cname = target_node.canonical_name
        new_value = target_node.active_value
        new_conf = float(target_node.confidence)
        new_cat = target_node.category
        new_pin = bool(target_node.pinned)
        new_name = target_node.canonical_name
        changes: List[str] = []

        # Confidence extraction (e.g. "99%", "0.98", "confidence to 95%")
        m_pct = re.search(r"(\d{1,3})\s*%", low)
        m_dec = re.search(r"confidence\s+(?:to\s+|=\s*)?(0\.\d+)", low)
        if m_pct:
            new_conf = max(0.0, min(1.0, float(m_pct.group(1)) / 100.0))
            changes.append(f"confidence → {int(new_conf * 100)}%")
        elif m_dec:
            new_conf = max(0.0, min(1.0, float(m_dec.group(1))))
            changes.append(f"confidence → {new_conf:.2f}")

        # Pin / unpin extraction
        if "unpin" in low:
            new_pin = False
            changes.append("unpinned")
        elif re.search(r"\bpin(?:ned)?\b", low):
            new_pin = True
            changes.append("★ pinned")

        # Category extraction
        for cat_candidate in ("architecture", "config", "decision", "tool", "constraint", "security", "task"):
            if re.search(rf"(?:category|type)\s+(?:to\s+|as\s+)?{cat_candidate}\b", low):
                new_cat = cat_candidate
                changes.append(f"category → {cat_candidate}")

        # Rename extraction
        m_ren = re.search(r"rename\s+(?:it\s+|this\s+|[^\s]+\s+)?to\s+([A-Za-z0-9_\-\s]{2,40}?)(?:\s+and\b|$)", raw, flags=re.IGNORECASE)
        if m_ren:
            cand_slug = "_".join(re.findall(r"[a-z0-9]+", m_ren.group(1).lower())[:4])
            if cand_slug:
                new_name = cand_slug
                changes.append(f"renamed → {new_name}")

        # Value / description extraction ("to say ...", "that ...", "value to ...", "is ...")
        for pattern in (
            r"(?:to\s+say|saying|description\s+to|value\s+to|belief\s+to)\s+[\"']?(.+?)[\"']?(?:\s+and\s+(?:set|pin|unpin|link|change)\b|$)",
            r"^(?:update|change|set|edit)\s+(?:this|it|[a-z0-9_\-\s]+?)\s+(?:to|as|=)\s+[\"']?(.+?)[\"']?(?:\s+and\s+(?:set|pin|unpin|link)\b|$)",
        ):
            m_val = re.search(pattern, raw, flags=re.IGNORECASE)
            if m_val:
                extracted_val = m_val.group(1).strip(" \"'.")
                if extracted_val and not extracted_val.lower().startswith(("confidence", "category", "pin")):
                    new_value = extracted_val
                    changes.append(f"belief → \"{new_value}\"")
                    break

        # If user just typed a descriptive sentence in a node-specific edit box without "update X to say"
        if not changes and default_node_name and cname == default_node_name and not low.endswith("?"):
            cleaned_val = re.sub(r"^(?:update|change|set|edit)\s+(?:this|it)?\s*(?:to\s+)?", "", raw, flags=re.IGNORECASE).strip(" \"'")
            if cleaned_val:
                new_value = cleaned_val
                changes.append(f"belief → \"{new_value}\"")

        # Link extraction ("link to X", "connect to X")
        edge_target = ""
        m_link = re.search(r"(?:link|connect)(?:\s+it|\s+this)?\s+to\s+([A-Za-z0-9_\-\s]+)$", raw, flags=re.IGNORECASE)
        if m_link:
            linked_node = _match_node_from_text(lake, m_link.group(1))
            if linked_node and linked_node.canonical_name != cname:
                edge_target = linked_node.canonical_name
                changes.append(f"linked ──► {edge_target}")

        if changes:
            c_info = summarize_concept(new_name, new_value, "", new_cat)
            tui.execute_key_action(
                "u",
                {
                    "node_id_or_name": cname,
                    "new_canonical_name": new_name if new_name != cname else None,
                    "category": new_cat,
                    "new_value": new_value,
                    "summary": f"{c_info['short_topic']}: {new_value}",
                    "confidence": new_conf,
                    "pinned": new_pin,
                    "edge_target": edge_target,
                    "edge_relation": "related_to",
                    "source_agent_id": "palace_nl_agent",
                    "provenance_type": "operator_crud",
                },
            )
            tui.selected_node_id = new_name
            summary_msg = f"Updated '{new_name}' ({c_info['short_topic']}): " + ", ".join(changes)
            tui.status_message = summary_msg
            tui.chat_history.append(
                {"user": raw, "palace": summary_msg, "cited_nodes": [new_name], "intent": "update"}
            )
            return {"status": "ok", "action": "update", "answer": summary_msg, "cited_nodes": [new_name]}

    # 5. Explain intent or general conversational query
    if target_node is not None and any(k in low for k in ("explain", "what is", "describe", "tell me about", "mean")):
        edges = lake.list_edges(node_id=target_node.node_id, active_only=True) if hasattr(lake, "list_edges") else []
        by_id = {n.node_id: n for n in (lake.list_concepts(include_tombstoned=False) if hasattr(lake, "list_concepts") else [])}
        exp = explain_concept_node(target_node, edges=edges, node_by_id=by_id)
        tui.selected_node_id = target_node.canonical_name
        ans = f"{exp['shape']} {exp['title']} ({target_node.canonical_name})\n• {exp['what_it_is']}\n• {exp['why_it_matters']}\n• {exp['origin']}"
        tui.status_message = f"Explained '{target_node.canonical_name}'"
        tui.chat_history.append(
            {"user": raw, "palace": ans, "cited_nodes": [target_node.canonical_name], "intent": "explain"}
        )
        return {"status": "ok", "action": "explain", "answer": ans, "cited_nodes": [target_node.canonical_name]}

    # Fallback to standard conversational ask
    return tui.handle_command(f"a {raw}")

