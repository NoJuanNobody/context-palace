"""
Context Palace v2 TUI view layer: monochrome "mint instrument" theme and Rich builders
for the WINDOWS / LAKE / GRAPH / AUDIT panes, the header and the hotkey status bar.

Design source: Designbot brief (`BRIEF.md`, `CP_tui.png`, `html/tui.html`, `html/tokens.css`).
The builders are pure functions of :class:`DashboardData`, which ``MemoryPalaceTUI`` fills
from the live lake/registry/retriever, so the same renderables back both
``context-palace tui --once`` (Rich, printed) and the interactive Textual app.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Design tokens (html/tokens.css). Terminals have no alpha, so the ink tints
# are pre-blended onto the sage background.
# ---------------------------------------------------------------------------
BG = "#e8efe6"
INK = "#1c1c1c"
INK_60 = "#6e706d"  # rgba(28,28,28,.60) on BG
INK_45 = "#8c908b"  # rgba(28,28,28,.45) on BG
INK_28 = "#afb4ad"  # rgba(28,28,28,.28) on BG
INK_16 = "#c7cdc6"  # rgba(28,28,28,.16) on BG  (rules)

S_INK = f"{INK} on {BG}"
S_60 = f"{INK_60} on {BG}"
S_45 = f"{INK_45} on {BG}"
S_28 = f"{INK_28} on {BG}"
S_LABEL = f"{INK_45} on {BG}"  # uppercase pane / column labels
S_STRONG = f"bold {INK} on {BG}"
S_PILL_ON = f"bold {BG} on {INK}"  # the only accent: solid ink fill

# Hairline-ruled table: no outer box, a rule under the header and between rows.
HAIRLINE_ROWS = box.Box(
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    " ── \n"
    " ── \n"
    "    \n"
    "    \n"
)

# Status bar hotkeys, in the order of the design: (key, label, action).
HOTKEYS: Tuple[Tuple[str, str], ...] = (
    ("c", "Create"),
    ("r", "Read"),
    ("u", "Update"),
    ("d", "Delete"),
    ("l", "List All"),
    ("a", "Ask"),
    ("p", "Pin & Inject"),
    ("m", "Models"),
    ("t", "Time-Travel"),
    ("q", "Quit"),
)

SPARK_CHARS = "▁▂▃▄▅▆▇█"
STACK_BELOW_COLUMNS = 160  # below this the 2x2 grid stacks into one column


@dataclass
class GraphRow:
    prefix: str
    relation: str
    name: str
    category: str
    confidence: Optional[float]
    pinned: bool
    provenance: str
    is_root: bool = False
    value: str = ""


@dataclass
class DashboardData:
    """Everything the panes need, collected from the live modules in one pass."""

    now: float
    as_of: Optional[float]
    db_path: str
    status_message: str = "Ready"
    search_query: str = ""
    selected_agent_id: Optional[str] = None
    sessions: List[Any] = field(default_factory=list)
    nodes: List[Any] = field(default_factory=list)
    edges: List[Any] = field(default_factory=list)
    conflicts: List[Any] = field(default_factory=list)
    node_provenance: Dict[str, str] = field(default_factory=dict)
    revision_provenance: Dict[str, str] = field(default_factory=dict)
    revision_count: Optional[int] = None
    revision_timestamps: List[float] = field(default_factory=list)
    provenance_weights: Dict[str, float] = field(default_factory=dict)
    last_capsule: Optional[Dict[str, Any]] = None
    last_output: List[str] = field(default_factory=list)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    frontend: str = "rich"
    size: Optional[Tuple[int, int]] = None

    @property
    def mode_label(self) -> str:
        if self.as_of is not None:
            return f"TIME-TRAVEL as_of={self.as_of:.2f}"
        return "LIVE"

    @property
    def node_by_id(self) -> Dict[str, Any]:
        return {n.node_id: n for n in self.nodes}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _clip(text: str, width: int) -> str:
    text = str(text)
    if width <= 0:
        return ""
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def _short_path(path: str) -> str:
    import os

    home = os.path.expanduser("~")
    if path and path.startswith(home):
        return "~" + path[len(home):]
    return path or ":memory:"


def pane_heading(label: str, hint: str = "") -> Table:
    """Uppercase pane label on the left, quiet hint on the right (SHELL / SYSTEM pattern)."""
    grid = Table.grid(expand=True, padding=(0, 2, 0, 0))
    grid.add_column(justify="left", no_wrap=True, min_width=len(label))
    grid.add_column(justify="right", no_wrap=True, overflow="ellipsis", ratio=1)
    grid.add_row(Text(label.upper(), style=S_INK), Text(hint, style=S_45))
    return grid


def usage_bar(used: int, maximum: int, width: int = 12) -> Text:
    ratio = 0.0 if maximum <= 0 else max(0.0, min(1.0, used / maximum))
    filled = int(round(ratio * width))
    if used > 0 and filled == 0:
        filled = 1
    bar = Text()
    bar.append("━" * filled, style=S_INK)
    bar.append("━" * (width - filled), style=S_28)
    return bar


def status_pill(needs_compaction: bool) -> Text:
    if needs_compaction:
        return Text(" COMPACT NOW ", style=S_PILL_ON)
    return Text("( HEALTHY )", style=S_INK)


def sparkline(timestamps: Sequence[float], end: float, buckets: int = 32, window_s: float = 6 * 3600) -> Text:
    """Hairline bar sparkline of belief-revision (ingest) activity."""
    stamps = [t for t in timestamps if t <= end]
    start = end - window_s
    if stamps and not any(t >= start for t in stamps):
        # Nothing in the recent window: span the whole history instead.
        start = min(stamps)
        window_s = max(1.0, end - start)
    counts = [0] * buckets
    for t in stamps:
        if t < start:
            continue
        idx = min(buckets - 1, int((t - start) / window_s * buckets))
        counts[idx] += 1
    peak = max(counts) if counts else 0
    out = Text()
    for c in counts:
        if peak == 0 or c == 0:
            out.append(SPARK_CHARS[0], style=S_28)
        else:
            level = max(1, math.ceil(c / peak * (len(SPARK_CHARS) - 1)))
            out.append(SPARK_CHARS[level], style=S_INK)
    return out


# ---------------------------------------------------------------------------
# header + status bar
# ---------------------------------------------------------------------------
def build_header(data: DashboardData) -> Table:
    clock = time.strftime("%H:%M", time.localtime(data.now))
    top = Table.grid(expand=True)
    top.add_column(justify="left")
    top.add_column(justify="right")
    top.add_row(Text(clock, style=S_60), Text("−  □  ×", style=S_INK))

    title = Text("Context Palace", style=S_STRONG)
    right = Text(justify="right")
    right.append(data.mode_label, style=S_STRONG)
    right.append("  ·  context-palace tui", style=S_60)
    right.append("\nMemoryPalaceTUI · Rich / Textual", style=S_60)
    if data.search_query:
        right.append(f"\nfilter '{data.search_query}'", style=S_45)
    if data.selected_agent_id:
        right.append(f"\nagent {data.selected_agent_id}", style=S_45)

    row = Table.grid(expand=True)
    row.add_column(justify="left", vertical="bottom")
    row.add_column(justify="right")
    row.add_row(title, right)

    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_row(top)
    grid.add_row(row)
    return grid


def build_status_bar(data: DashboardData, width: Optional[int] = None) -> Table:
    left = Text()
    left.append("LIVE" if data.as_of is None else "TIME-TRAVEL", style=S_INK)
    left.append("  │  ", style=S_28)
    db_label = _short_path(data.db_path)
    if len(db_label) > 40:
        db_label = "…" + db_label[-39:]
    left.append(db_label, style=S_60)

    keys = Text()
    for i, (key, label) in enumerate(HOTKEYS):
        if i:
            keys.append(" ")
        keys.append("[", style=S_45)
        keys.append(key, style=S_STRONG)
        keys.append("] ", style=S_45)
        keys.append(label, style=S_INK)

    right = Text(justify="right")
    right.append(data.frontend, style=S_60)
    if data.size:
        right.append("  │  ", style=S_28)
        right.append(f"{data.size[0]}×{data.size[1]}", style=S_60)
    right.append("  │  ", style=S_28)
    right.append("UTF-8", style=S_60)

    if width is not None and width < STACK_BELOW_COLUMNS:
        # Narrow: hotkeys on their own line so no "[x] Label" pair is split.
        meta = Table.grid(expand=True)
        meta.add_column(justify="left", no_wrap=True, overflow="ellipsis")
        meta.add_column(justify="right", no_wrap=True)
        meta.add_row(left, right)
        stacked = Table.grid(expand=True)
        stacked.add_column(justify="left", no_wrap=True, overflow="ellipsis")
        stacked.add_row(keys)
        stacked.add_row(meta)
        return stacked

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left", no_wrap=True, overflow="ellipsis")
    grid.add_column(justify="center", no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(left, keys, right)
    return grid


# ---------------------------------------------------------------------------
# WINDOWS
# ---------------------------------------------------------------------------
def build_windows_pane(data: DashboardData) -> Group:
    table = Table(
        box=HAIRLINE_ROWS,
        show_header=True,
        show_lines=True,
        show_edge=False,
        expand=True,
        border_style=S_28,
        header_style=S_LABEL,
        pad_edge=False,
        padding=(0, 1),
        style=S_INK,
    )
    table.add_column("AGENT", no_wrap=True, overflow="ellipsis", ratio=3, min_width=10)
    table.add_column("HARNESS", no_wrap=True, overflow="ellipsis", ratio=2, style=S_60)
    table.add_column("MODEL", no_wrap=True, overflow="ellipsis", ratio=3, style=S_60)
    table.add_column("USED / MAX", no_wrap=True, ratio=4, min_width=20)
    table.add_column("HEADROOM", justify="right", no_wrap=True)
    table.add_column("BUDGET", justify="right", no_wrap=True)
    table.add_column("STATUS", justify="left", no_wrap=True, min_width=13)

    if not data.sessions:
        table.add_row(Text("(no agents registered)", style=S_45), "-", "-", Text("0 / 0", style=S_45), "0", "0", Text("IDLE", style=S_45))
    for sess in data.sessions:
        used = Text(f"{sess.used_tokens:,} / {sess.max_context_tokens:,}", style=S_INK)
        bar = usage_bar(sess.used_tokens, sess.max_context_tokens)
        bar.append(f"  {sess.utilization_pct:.2f}%", style=S_INK)
        name_style = S_STRONG if sess.agent_id == data.selected_agent_id else S_INK
        table.add_row(
            Text(sess.agent_id, style=name_style),
            sess.harness_type,
            sess.model_id,
            Group(used, bar),
            f"{sess.remaining_headroom:,}",
            f"{sess.adaptive_capsule_budget():,}",
            status_pill(sess.needs_compaction),
        )

    note = Text("COMPACT NOW flags ≥ 80% utilization · indicator only, no compaction action is wired yet", style=S_45)
    return Group(pane_heading("Windows", "[m] refresh model windows"), table, Text(""), note)


# ---------------------------------------------------------------------------
# LAKE
# ---------------------------------------------------------------------------
def build_lake_pane(data: DashboardData) -> Group:
    import os

    lake_dir = _short_path(os.path.dirname(data.db_path)) if data.db_path not in ("", ":memory:") else ":memory:"
    if len(lake_dir) > 28:
        lake_dir = "…" + lake_dir[-27:]
    counts = [
        ("CONCEPTS", len(data.nodes)),
        ("PINNED", sum(1 for n in data.nodes if getattr(n, "pinned", False))),
        ("EDGES", len(data.edges)),
        ("CONFLICTS", len(data.conflicts)),
    ]
    nums = Table.grid(expand=True)
    for _ in counts:
        nums.add_column(ratio=1)
    nums.add_row(*[Text(k, style=S_LABEL) for k, _ in counts])
    nums.add_row(*[Text(f"{v:,}", style=S_STRONG) for _, v in counts])

    # Tier distribution (definitions from the v2 FigCraft TUI): HOT = pinned or conf >= 0.85,
    # WARM = 0.50 <= conf < 0.85, COLD = conf < 0.50, SUPERSEDED = tombstoned.
    active = [n for n in data.nodes if not getattr(n, "tombstoned", False)]
    tiers = (
        ("HOT", sum(1 for n in active if n.pinned or n.confidence >= 0.85)),
        ("WARM", sum(1 for n in active if not n.pinned and 0.50 <= n.confidence < 0.85)),
        ("COLD", sum(1 for n in active if not n.pinned and n.confidence < 0.50)),
        ("SUPERSEDED", sum(1 for n in data.nodes if getattr(n, "tombstoned", False))),
    )
    tier_line = Text("TIERS  ", style=S_LABEL)
    for i, (name, count) in enumerate(tiers):
        if i:
            tier_line.append("  ·  ", style=S_28)
        tier_line.append(f"{name.lower()} ", style=S_45)
        tier_line.append(f"{count:,}", style=S_INK)

    end = data.as_of if data.as_of is not None else data.now
    spark = sparkline(data.revision_timestamps, end=end)
    rev_note = Text(
        f"belief revisions{'' if data.revision_count is None else f' {data.revision_count:,}'} · ingest activity, last 6h",
        style=S_45,
    )

    wm = Table(
        box=HAIRLINE_ROWS,
        show_header=True,
        show_lines=True,
        show_edge=False,
        expand=True,
        border_style=S_28,
        header_style=S_LABEL,
        pad_edge=False,
        padding=(0, 1),
        style=S_INK,
    )
    wm.add_column("SESSION", no_wrap=True, overflow="ellipsis")
    wm.add_column("WATERMARK", no_wrap=True)
    wm.add_column("OFFSET", justify="right", no_wrap=True)
    wm.add_column("CAPSULE BUDGET", justify="right", no_wrap=True)
    if not data.sessions:
        wm.add_row(Text("(none)", style=S_45), "-", "-", "-")
    for sess in data.sessions:
        wm.add_row(
            sess.agent_id,
            f"turn {sess.last_processed_turn}",
            f"{sess.last_byte_offset:,}",
            f"{sess.adaptive_capsule_budget():,} tok",
        )

    prov = Text("provenance  ", style=S_LABEL)
    ordered = sorted(data.provenance_weights.items(), key=lambda kv: -kv[1])
    for i, (name, weight) in enumerate(ordered):
        if i:
            prov.append(" > ", style=S_45)
        prov.append(f"{name} {weight:.2f}", style=S_60)

    return Group(
        pane_heading("Lake", f"SQLite WAL · {lake_dir}"),
        nums,
        tier_line,
        spark,
        rev_note,
        wm,
        Text(""),
        prov,
    )


# ---------------------------------------------------------------------------
# GRAPH
# ---------------------------------------------------------------------------
def collect_graph_rows(data: DashboardData, max_roots: int = 8, max_children: int = 4, max_grandchildren: int = 2) -> List[GraphRow]:
    nodes = data.nodes
    by_id = data.node_by_id
    out_edges: Dict[str, List[Any]] = {}
    for e in data.edges:
        out_edges.setdefault(e.source_id, []).append(e)

    roots = sorted(
        nodes,
        key=lambda n: (not bool(n.pinned), -float(n.confidence), -float(getattr(n, "updated_at", 0.0)), n.canonical_name),
    )[:max_roots]

    def node_row(node: Any, prefix: str, relation: str, is_root: bool) -> GraphRow:
        return GraphRow(
            prefix=prefix,
            relation=relation,
            name=node.canonical_name + ("/" if is_root else ""),
            category=node.category,
            confidence=float(node.confidence),
            pinned=bool(node.pinned),
            provenance=data.node_provenance.get(node.node_id, ""),
            is_root=is_root,
            value=node.active_value if is_root else "",
        )

    rows: List[GraphRow] = []
    for root in roots:
        rows.append(node_row(root, "", "", True))
        children = out_edges.get(root.node_id, [])[:max_children]
        for ci, edge in enumerate(children):
            last = ci == len(children) - 1
            target = by_id.get(edge.target_id)
            branch = "└─ " if last else "├─ "
            if target is None:
                rows.append(GraphRow(branch, edge.relation_type, edge.target_id, "", None, False, ""))
                continue
            rows.append(node_row(target, branch, edge.relation_type, False))
            grand = [g for g in out_edges.get(target.node_id, []) if g.target_id != root.node_id][:max_grandchildren]
            for gi, gedge in enumerate(grand):
                gtarget = by_id.get(gedge.target_id)
                gbranch = ("   " if last else "│  ") + ("└─ " if gi == len(grand) - 1 else "├─ ")
                if gtarget is None:
                    rows.append(GraphRow(gbranch, gedge.relation_type, gedge.target_id, "", None, False, ""))
                else:
                    rows.append(node_row(gtarget, gbranch, gedge.relation_type, False))
    return rows


def build_graph_pane(data: DashboardData, max_roots: int = 8) -> Group:
    rows = collect_graph_rows(data, max_roots=max_roots)
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(no_wrap=True, overflow="ellipsis", ratio=5)  # tree + name
    table.add_column(no_wrap=True, overflow="ellipsis", ratio=2)  # category
    table.add_column(no_wrap=True)  # conf
    table.add_column(no_wrap=True, min_width=6)  # PINNED
    table.add_column(no_wrap=True, overflow="ellipsis", ratio=2)  # provenance

    if not rows:
        table.add_row(Text("(no active concepts)", style=S_45), "", "", "", "")
    for r in rows:
        name = Text()
        name.append(r.prefix, style=S_45)
        if r.relation:
            name.append(f"{r.relation}  ", style=S_60)
        name.append(r.name, style=S_STRONG if r.is_root else S_INK)
        table.add_row(
            name,
            Text(r.category, style=S_60),
            Text("" if r.confidence is None else f"conf {r.confidence:.2f}", style=S_60),
            Text("PINNED" if r.pinned else "", style=S_STRONG),
            Text(r.provenance, style=S_60),
        )
        if r.is_root and r.value and (r.pinned or data.search_query or data.as_of is not None):
            table.add_row(Text(f"   = {r.value}", style=S_45, overflow="fold"), "", "", "", "")

    pinned = sum(1 for n in data.nodes if n.pinned)
    shown = sum(1 for r in rows if r.is_root)
    footer = Text(
        f"{pinned:,} pinned · {len(data.nodes):,} listed"
        + (f" · showing {shown}, scroll for list-all" if shown < len(data.nodes) else ""),
        style=S_45,
    )
    return Group(
        pane_heading("Graph", "1-hop neighbourhood · pinned first"),
        table,
        Text(""),
        footer,
    )


# ---------------------------------------------------------------------------
# AUDIT
# ---------------------------------------------------------------------------
def build_audit_pane(data: DashboardData, max_events: int = 3, max_chat_turns: int = 2) -> Group:
    by_id = data.node_by_id
    lines: List[RenderableType] = [pane_heading("Audit", "consensus · last capsule · [a] ask")]
    lines.append(Text("ConflictResolutionEvent", style=S_INK))
    events = sorted(data.conflicts, key=lambda e: -float(getattr(e, "resolved_at", 0.0)))[:max_events]
    if not events:
        lines.append(Text("  0 conflicts resolved", style=S_45))
    for ev in events:
        node = by_id.get(ev.node_id)
        head = Text()
        head.append(_clip(ev.event_id, 18), style=S_INK)
        head.append("  ")
        head.append(node.canonical_name if node else ev.node_id, style=S_STRONG)
        lines.append(head)
        for label, agent, score, rev_id in (
            ("winner", ev.winning_agent_id, ev.winning_score, ev.winning_revision_id),
            ("loser ", ev.losing_agent_id, ev.losing_score, ev.losing_revision_id),
        ):
            t = Text("  ")
            t.append(label, style=S_45)
            t.append(f" {_clip(agent, 22):<22} ", style=S_INK)
            t.append(f"{score:.3f}  ", style=S_INK)
            t.append(data.revision_provenance.get(rev_id, ""), style=S_60)
            lines.append(t)
        if ev.reason:
            t = Text("  ")
            t.append("reason", style=S_45)
            t.append(f" {ev.reason}", style=S_60)
            lines.append(t)
    lines.append(Text(""))

    cap = data.last_capsule
    if cap:
        head = Text("context_inject_capsule  ", style=S_INK)
        head.append(str(cap.get("target_agent_id", "")), style=S_STRONG)
        lines.append(head)
        sess = next((s for s in data.sessions if s.agent_id == cap.get("target_agent_id")), None)
        if sess is not None:
            t = Text(f"  window {sess.used_tokens:,} / {sess.max_context_tokens:,} ({sess.utilization_pct:.2f}%)  ", style=S_INK)
            t.append_text(status_pill(sess.needs_compaction))
            lines.append(t)
        lines.append(
            Text(
                f"  headroom {int(cap.get('remaining_headroom', 0)):,}   budget {int(cap.get('allocated_budget_tokens', 0)):,}"
                f"   capsule {int(cap.get('capsule_tokens', 0)):,}",
                style=S_INK,
            )
        )
        lines.append(
            Text(
                f"  raw {int(cap.get('raw_history_tokens_equivalent', 0)):,}   reduction {float(cap.get('token_reduction_pct', 0.0)):.2f}%",
                style=S_INK,
            )
        )
        if cap.get("sidecar_path"):
            lines.append(Text(f"  sidecar {cap['sidecar_path']}", style=S_60))
    else:
        t = Text("context_inject_capsule  ", style=S_INK)
        t.append("no capsule pushed this session · [p] Pin & Inject", style=S_45)
        lines.append(t)
    lines.append(Text(""))

    # Conversational engine ([a] Ask): last turns, answered from the lake.
    if data.chat_history:
        for turn in data.chat_history[-max_chat_turns:]:
            you = Text("You:    ", style=S_45)
            you.append(str(turn.get("user", "")), style=S_INK)
            lines.append(you)
            answer_lines = str(turn.get("palace", "")).splitlines() or [""]
            for i, ans in enumerate(answer_lines[:5]):
                t = Text("Palace: " if i == 0 else "        ", style=S_45)
                t.append(ans, style=S_INK)
                lines.append(t)
            if len(answer_lines) > 5:
                lines.append(Text(f"        … {len(answer_lines) - 5} more", style=S_45))
            cited = turn.get("cited_nodes") or []
            if cited:
                lines.append(Text(f"        └─ cited: {', '.join(map(str, cited[:6]))}", style=S_60))
        lines.append(Text(""))
    else:
        lines.append(Text("[a] Ask the palace · e.g. 'What do my agents know about auth?', 'Remember port is 8443'", style=S_45))
        lines.append(Text(""))

    for line in data.last_output[-6:]:
        lines.append(Text(line, style=S_60, overflow="fold"))
    prompt = Text("palace$ ", style=S_INK)
    prompt.append(data.status_message, style=S_60)
    prompt.append(" █", style=S_INK)
    lines.append(prompt)
    return Group(*lines)


# ---------------------------------------------------------------------------
# full static render (context-palace tui --once)
# ---------------------------------------------------------------------------
def _pane(renderable: RenderableType) -> Panel:
    return Panel(renderable, box=box.ROUNDED, border_style=S_INK, style=S_INK, padding=(0, 1))


def build_dashboard(data: DashboardData, width: int) -> Panel:
    windows = _pane(build_windows_pane(data))
    lake = _pane(build_lake_pane(data))
    graph = _pane(build_graph_pane(data))
    audit = _pane(build_audit_pane(data))

    body = Table.grid(expand=True, padding=(0, 1))
    if width >= STACK_BELOW_COLUMNS:
        body.add_column(ratio=125)
        body.add_column(ratio=75)
        body.add_row(windows, lake)
        body.add_row(graph, audit)
    else:
        body.add_column()
        for pane in (windows, lake, graph, audit):
            body.add_row(pane)

    rule = Rule(style=S_28, characters="─")
    window = Group(build_header(data), Text(""), body, rule, build_status_bar(data, width=width))
    return Panel(window, box=box.ROUNDED, border_style=S_INK, style=S_INK, padding=(0, 1))
