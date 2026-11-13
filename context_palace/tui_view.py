"""
Context Palace v2 TUI view layer: dual-mode ("Obsidian Mint" dark + "Sage Mint" light)
instrument theme, clickable Rich action spans, and builders for the
WINDOWS / LAKE / GRAPH / AUDIT panes, the header, and the hotkey status bar.

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
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Design tokens (Light "Sage Mint Instrument" & Dark "Obsidian Mint Instrument")
# ---------------------------------------------------------------------------
LIGHT_BG = "#e8efe6"
LIGHT_INK = "#1c1c1c"
LIGHT_INK_60 = "#6e706d"
LIGHT_INK_45 = "#8c908b"
LIGHT_INK_28 = "#afb4ad"
LIGHT_INK_16 = "#c7cdc6"

DARK_BG = "#0d1210"
DARK_INK = "#d8e6df"
DARK_INK_60 = "#8fa39a"
DARK_INK_45 = "#687a72"
DARK_INK_28 = "#3d4d46"
DARK_INK_16 = "#232e29"

# Backwards-compatible module constants
BG = LIGHT_BG
INK = LIGHT_INK
INK_60 = LIGHT_INK_60
INK_45 = LIGHT_INK_45
INK_28 = LIGHT_INK_28
INK_16 = LIGHT_INK_16

S_INK = f"{INK} on {BG}"
S_60 = f"{INK_60} on {BG}"
S_45 = f"{INK_45} on {BG}"
S_28 = f"{INK_28} on {BG}"
S_LABEL = f"{INK_45} on {BG}"
S_STRONG = f"bold {INK} on {BG}"
S_PILL_ON = f"bold {BG} on {INK}"


@dataclass(frozen=True)
class ThemePalette:
    mode: str
    bg: str
    ink: str
    ink_60: str
    ink_45: str
    ink_28: str
    ink_16: str

    @property
    def s_ink(self) -> str:
        return f"{self.ink} on {self.bg}"

    @property
    def s_60(self) -> str:
        return f"{self.ink_60} on {self.bg}"

    @property
    def s_45(self) -> str:
        return f"{self.ink_45} on {self.bg}"

    @property
    def s_28(self) -> str:
        return f"{self.ink_28} on {self.bg}"

    @property
    def s_label(self) -> str:
        return f"{self.ink_45} on {self.bg}"

    @property
    def s_strong(self) -> str:
        return f"bold {self.ink} on {self.bg}"

    @property
    def s_pill_on(self) -> str:
        return f"bold {self.bg} on {self.ink}"


THEMES: Dict[str, ThemePalette] = {
    "dark": ThemePalette(
        mode="dark",
        bg=DARK_BG,
        ink=DARK_INK,
        ink_60=DARK_INK_60,
        ink_45=DARK_INK_45,
        ink_28=DARK_INK_28,
        ink_16=DARK_INK_16,
    ),
    "light": ThemePalette(
        mode="light",
        bg=LIGHT_BG,
        ink=LIGHT_INK,
        ink_60=LIGHT_INK_60,
        ink_45=LIGHT_INK_45,
        ink_28=LIGHT_INK_28,
        ink_16=LIGHT_INK_16,
    ),
}


def get_theme(mode: Optional[str] = None) -> ThemePalette:
    if mode and mode.lower() in THEMES:
        return THEMES[mode.lower()]
    return THEMES["dark"]


def _click(base_style: str, action: str, *args: Any) -> Style:
    """Attach a Textual @click action meta tag to a Rich style string."""
    meta = {"@click": (f"app.{action}", tuple(args)) if args else f"app.{action}"}
    return Style.parse(base_style) + Style(meta=meta)


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

# Status bar hotkeys, in the order of the design: (key, label).
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
    ("k", "Theme"),
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
    canonical_name: str = ""


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
    theme_mode: str = "dark"

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


def pane_heading(
    label: str,
    hint: Any = "",
    theme: Optional[ThemePalette] = None,
    label_action: Optional[str] = None,
) -> Table:
    """Uppercase pane label on the left, quiet/clickable hint on the right."""
    th = theme or get_theme()
    grid = Table.grid(expand=True, padding=(0, 2, 0, 0))
    grid.add_column(justify="left", no_wrap=True, min_width=len(label))
    grid.add_column(justify="right", no_wrap=True, overflow="ellipsis", ratio=1)
    lbl_style: Any = _click(th.s_ink, label_action) if label_action else th.s_ink
    right_cell = hint if isinstance(hint, Text) else Text(str(hint), style=th.s_45)
    grid.add_row(Text(label.upper(), style=lbl_style), right_cell)
    return grid


def usage_bar(used: int, maximum: int, width: int = 12, theme: Optional[ThemePalette] = None) -> Text:
    th = theme or get_theme()
    ratio = 0.0 if maximum <= 0 else max(0.0, min(1.0, used / maximum))
    filled = int(round(ratio * width))
    if used > 0 and filled == 0:
        filled = 1
    bar = Text()
    bar.append("━" * filled, style=th.s_ink)
    bar.append("━" * (width - filled), style=th.s_28)
    return bar


def status_pill(
    needs_compaction: bool,
    theme: Optional[ThemePalette] = None,
    agent_id: Optional[str] = None,
) -> Text:
    th = theme or get_theme()
    if needs_compaction:
        st: Any = _click(th.s_pill_on, "compact_agent", agent_id) if agent_id else th.s_pill_on
        return Text(" COMPACT NOW ", style=st)
    st = _click(th.s_ink, "manage_agent", agent_id) if agent_id else th.s_ink
    return Text("( HEALTHY )", style=st)


def sparkline(
    timestamps: Sequence[float],
    end: float,
    buckets: int = 32,
    window_s: float = 6 * 3600,
    theme: Optional[ThemePalette] = None,
) -> Text:
    """Hairline bar sparkline of belief-revision (ingest) activity."""
    th = theme or get_theme()
    stamps = [t for t in timestamps if t <= end]
    start = end - window_s
    if stamps and not any(t >= start for t in stamps):
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
            out.append(SPARK_CHARS[0], style=th.s_28)
        else:
            level = max(1, math.ceil(c / peak * (len(SPARK_CHARS) - 1)))
            out.append(SPARK_CHARS[level], style=th.s_ink)
    return out


# ---------------------------------------------------------------------------
# header + status bar
# ---------------------------------------------------------------------------
def build_header(data: DashboardData) -> Table:
    th = get_theme(data.theme_mode)
    clock = time.strftime("%H:%M", time.localtime(data.now))
    top = Table.grid(expand=True)
    top.add_column(justify="left")
    top.add_column(justify="right")

    left_top = Text(clock, style=th.s_60)
    left_top.append("  ")
    theme_badge = "☾ DARK" if th.mode == "dark" else "☀ LIGHT"
    left_top.append(f"[ {theme_badge} ]", style=_click(th.s_pill_on, "toggle_theme"))
    left_top.append("  ")
    left_top.append("[ + New Concept ]", style=_click(th.s_strong, "create"))
    left_top.append("  ")
    left_top.append("[ ✎ CRUD Manager ]", style=_click(th.s_strong, "open_crud_hub"))

    win_ctrl = Text()
    win_ctrl.append("−", style=_click(th.s_ink, "clear_filter"))
    win_ctrl.append("  ", style=th.s_ink)
    win_ctrl.append("□", style=_click(th.s_ink, "list_all"))
    win_ctrl.append("  ", style=th.s_ink)
    win_ctrl.append("×", style=_click(th.s_ink, "quit"))
    top.add_row(left_top, win_ctrl)

    title = Text("Context Palace", style=_click(th.s_strong, "clear_filter"))
    right = Text(justify="right")
    right.append(data.mode_label, style=_click(th.s_strong, "time_travel"))
    right.append("  ·  context-palace tui", style=th.s_60)
    right.append(f"\nMemoryPalaceTUI · Rich / Textual · {th.mode.upper()}", style=th.s_60)
    if data.search_query:
        right.append(f"\nfilter '{data.search_query}' ", style=th.s_45)
        right.append("[clear ×]", style=_click(th.s_strong, "clear_filter"))
    if data.selected_agent_id:
        right.append(f"\nagent {data.selected_agent_id} ", style=th.s_45)
        right.append("[manage]", style=_click(th.s_strong, "manage_agent", data.selected_agent_id))

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
    th = get_theme(data.theme_mode)
    left = Text()
    left.append(
        "LIVE" if data.as_of is None else "TIME-TRAVEL",
        style=_click(th.s_ink, "time_travel"),
    )
    left.append("  │  ", style=th.s_28)
    db_label = _short_path(data.db_path)
    if len(db_label) > 34:
        db_label = "…" + db_label[-33:]
    left.append(db_label, style=_click(th.s_60, "list_all"))

    keys = Text()
    for i, (key, label) in enumerate(HOTKEYS):
        if i:
            keys.append(" ")
        keys.append("[", style=_click(th.s_45, "hotkey", key))
        keys.append(key, style=_click(th.s_strong, "hotkey", key))
        keys.append("] ", style=_click(th.s_45, "hotkey", key))
        keys.append(label, style=_click(th.s_ink, "hotkey", key))

    right = Text(justify="right")
    right.append(data.frontend, style=th.s_60)
    if data.size:
        right.append("  │  ", style=th.s_28)
        right.append(f"{data.size[0]}×{data.size[1]}", style=th.s_60)
    right.append("  │  ", style=th.s_28)
    right.append("UTF-8", style=th.s_60)

    if width is not None and width < STACK_BELOW_COLUMNS:
        meta = Table.grid(expand=True)
        meta.add_column(justify="left", no_wrap=True, overflow="ellipsis")
        meta.add_column(justify="right", no_wrap=True)
        meta.add_row(left, right)
        stacked = Table.grid(expand=True)
        stacked.add_column(justify="left", no_wrap=True, overflow="ellipsis")
        stacked.add_row(keys)
        stacked.add_row(meta)
        return stacked

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(justify="left", no_wrap=True, overflow="ellipsis")
    grid.add_column(justify="center", no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(left, keys, right)
    return grid


# ---------------------------------------------------------------------------
# WINDOWS
# ---------------------------------------------------------------------------
def build_windows_pane(data: DashboardData) -> Group:
    th = get_theme(data.theme_mode)
    table = Table(
        box=HAIRLINE_ROWS,
        show_header=True,
        show_lines=True,
        show_edge=False,
        expand=True,
        border_style=th.s_28,
        header_style=th.s_label,
        pad_edge=False,
        padding=(0, 1),
        style=th.s_ink,
    )
    table.add_column("AGENT", no_wrap=True, overflow="ellipsis", ratio=3, min_width=10)
    table.add_column("HARNESS", no_wrap=True, overflow="ellipsis", ratio=2, style=th.s_60)
    table.add_column("MODEL", no_wrap=True, overflow="ellipsis", ratio=3, style=th.s_60)
    table.add_column("USED / MAX", no_wrap=True, ratio=4, min_width=20)
    table.add_column("HEADROOM", justify="right", no_wrap=True)
    table.add_column("BUDGET", justify="right", no_wrap=True)
    table.add_column("STATUS", justify="left", no_wrap=True, min_width=13)

    if not data.sessions:
        table.add_row(
            Text("(no agents registered)", style=_click(th.s_45, "manage_agent", "")),
            "-",
            "-",
            Text("0 / 0", style=th.s_45),
            "0",
            "0",
            Text("IDLE", style=th.s_45),
        )
    for sess in data.sessions:
        aid = sess.agent_id
        used = Text(f"{sess.used_tokens:,} / {sess.max_context_tokens:,}", style=_click(th.s_ink, "manage_agent", aid))
        bar = usage_bar(sess.used_tokens, sess.max_context_tokens, theme=th)
        bar.append(f"  {sess.utilization_pct:.2f}%", style=_click(th.s_ink, "manage_agent", aid))
        name_base = th.s_strong if aid == data.selected_agent_id else th.s_ink
        table.add_row(
            Text(aid, style=_click(name_base, "manage_agent", aid)),
            Text(str(sess.harness_type), style=_click(th.s_60, "manage_agent", aid)),
            Text(str(sess.model_id), style=_click(th.s_60, "manage_agent", aid)),
            Group(used, bar),
            Text(f"{sess.remaining_headroom:,}", style=_click(th.s_ink, "manage_agent", aid)),
            Text(f"{sess.adaptive_capsule_budget():,}", style=_click(th.s_ink, "manage_agent", aid)),
            status_pill(sess.needs_compaction, theme=th, agent_id=aid),
        )

    hint = Text()
    hint.append("[+ agent]", style=_click(th.s_strong, "manage_agent", ""))
    hint.append(" · ", style=th.s_45)
    hint.append("[m] refresh model windows", style=_click(th.s_45, "models"))

    note = Text("Click any agent row to edit/delete session · click ", style=th.s_45)
    note.append("COMPACT NOW", style=_click(th.s_strong, "pin_inject"))
    note.append(" to inject compacted capsule", style=th.s_45)
    return Group(pane_heading("Windows", hint, theme=th, label_action="models"), table, Text(""), note)


# ---------------------------------------------------------------------------
# LAKE
# ---------------------------------------------------------------------------
def build_lake_pane(data: DashboardData) -> Group:
    import os

    th = get_theme(data.theme_mode)
    lake_dir = _short_path(os.path.dirname(data.db_path)) if data.db_path not in ("", ":memory:") else ":memory:"
    if len(lake_dir) > 24:
        lake_dir = "…" + lake_dir[-23:]
    counts = [
        ("CONCEPTS", len(data.nodes), "clear_filter", ()),
        ("PINNED", sum(1 for n in data.nodes if getattr(n, "pinned", False)), "filter_pinned", ()),
        ("EDGES", len(data.edges), "list_all", ()),
        ("CONFLICTS", len(data.conflicts), "list_all", ()),
    ]
    nums = Table.grid(expand=True)
    for _ in counts:
        nums.add_column(ratio=1)
    nums.add_row(*[Text(k, style=_click(th.s_label, act, *args)) for k, _, act, args in counts])
    nums.add_row(*[Text(f"{v:,}", style=_click(th.s_strong, act, *args)) for _, v, act, args in counts])

    active = [n for n in data.nodes if not getattr(n, "tombstoned", False)]
    tiers = (
        ("HOT", sum(1 for n in active if n.pinned or n.confidence >= 0.85)),
        ("WARM", sum(1 for n in active if not n.pinned and 0.50 <= n.confidence < 0.85)),
        ("COLD", sum(1 for n in active if not n.pinned and n.confidence < 0.50)),
        ("SUPERSEDED", sum(1 for n in data.nodes if getattr(n, "tombstoned", False))),
    )
    tier_line = Text("TIERS  ", style=th.s_label)
    for i, (name, count) in enumerate(tiers):
        if i:
            tier_line.append("  ·  ", style=th.s_28)
        tier_line.append(f"{name.lower()} ", style=_click(th.s_45, "list_all"))
        tier_line.append(f"{count:,}", style=_click(th.s_ink, "list_all"))

    end = data.as_of if data.as_of is not None else data.now
    spark = sparkline(data.revision_timestamps, end=end, theme=th)
    rev_note = Text(
        f"belief revisions{'' if data.revision_count is None else f' {data.revision_count:,}'} · ingest activity, last 6h",
        style=_click(th.s_45, "time_travel"),
    )

    wm = Table(
        box=HAIRLINE_ROWS,
        show_header=True,
        show_lines=True,
        show_edge=False,
        expand=True,
        border_style=th.s_28,
        header_style=th.s_label,
        pad_edge=False,
        padding=(0, 1),
        style=th.s_ink,
    )
    wm.add_column("SESSION", no_wrap=True, overflow="ellipsis")
    wm.add_column("WATERMARK", no_wrap=True)
    wm.add_column("OFFSET", justify="right", no_wrap=True)
    wm.add_column("CAPSULE BUDGET", justify="right", no_wrap=True)
    if not data.sessions:
        wm.add_row(Text("(none)", style=th.s_45), "-", "-", "-")
    for sess in data.sessions:
        aid = sess.agent_id
        wm.add_row(
            Text(aid, style=_click(th.s_ink, "manage_agent", aid)),
            Text(f"turn {sess.last_processed_turn}", style=_click(th.s_ink, "manage_agent", aid)),
            Text(f"{sess.last_byte_offset:,}", style=_click(th.s_ink, "manage_agent", aid)),
            Text(f"{sess.adaptive_capsule_budget():,} tok", style=_click(th.s_ink, "compact_agent", aid)),
        )

    prov = Text("provenance  ", style=th.s_label)
    ordered = sorted(data.provenance_weights.items(), key=lambda kv: -kv[1])
    for i, (name, weight) in enumerate(ordered):
        if i:
            prov.append(" > ", style=th.s_45)
        prov.append(f"{name} {weight:.2f}", style=_click(th.s_60, "filter_concepts", name))

    lake_hint = Text()
    lake_hint.append("[l] list all", style=_click(th.s_strong, "list_all"))
    lake_hint.append(f" · SQLite WAL · {lake_dir}", style=th.s_45)

    return Group(
        pane_heading("Lake", lake_hint, theme=th, label_action="list_all"),
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
def collect_graph_rows(
    data: DashboardData, max_roots: int = 8, max_children: int = 4, max_grandchildren: int = 2
) -> List[GraphRow]:
    nodes = data.nodes
    by_id = data.node_by_id
    out_edges: Dict[str, List[Any]] = {}
    for e in data.edges:
        out_edges.setdefault(e.source_id, []).append(e)

    roots = sorted(
        nodes,
        key=lambda n: (
            not bool(n.pinned),
            -float(n.confidence),
            -float(getattr(n, "updated_at", 0.0)),
            n.canonical_name,
        ),
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
            canonical_name=node.canonical_name,
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
                rows.append(
                    GraphRow(
                        branch,
                        edge.relation_type,
                        edge.target_id,
                        "",
                        None,
                        False,
                        "",
                        canonical_name=edge.target_id,
                    )
                )
                continue
            rows.append(node_row(target, branch, edge.relation_type, False))
            grand = [g for g in out_edges.get(target.node_id, []) if g.target_id != root.node_id][
                :max_grandchildren
            ]
            for gi, gedge in enumerate(grand):
                gtarget = by_id.get(gedge.target_id)
                gbranch = ("   " if last else "│  ") + ("└─ " if gi == len(grand) - 1 else "├─ ")
                if gtarget is None:
                    rows.append(
                        GraphRow(
                            gbranch,
                            gedge.relation_type,
                            gedge.target_id,
                            "",
                            None,
                            False,
                            "",
                            canonical_name=gedge.target_id,
                        )
                    )
                else:
                    rows.append(node_row(gtarget, gbranch, gedge.relation_type, False))
    return rows


def build_graph_pane(data: DashboardData, max_roots: int = 8) -> Group:
    th = get_theme(data.theme_mode)
    rows = collect_graph_rows(data, max_roots=max_roots)
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(no_wrap=True, overflow="ellipsis", ratio=5)  # tree + name
    table.add_column(no_wrap=True, overflow="ellipsis", ratio=2)  # category
    table.add_column(no_wrap=True)  # conf
    table.add_column(no_wrap=True, min_width=6)  # PINNED
    table.add_column(no_wrap=True, overflow="ellipsis", ratio=2)  # provenance + quick actions

    if not rows:
        table.add_row(
            Text("(no active concepts)", style=_click(th.s_45, "create")),
            "",
            "",
            "",
            "",
        )
    for r in rows:
        cname = r.canonical_name or r.name.rstrip("/")
        name = Text()
        name.append(r.prefix, style=th.s_45)
        if r.relation:
            name.append(f"{r.relation}  ", style=th.s_60)
        name_style = th.s_strong if r.is_root else th.s_ink
        name.append(r.name, style=_click(name_style, "open_concept_crud", cname))

        pin_cell = Text()
        if r.pinned:
            pin_cell.append("PINNED", style=_click(th.s_strong, "quick_toggle_pin", cname))
        else:
            pin_cell.append("[pin]", style=_click(th.s_45, "quick_toggle_pin", cname))

        prov_cell = Text()
        if r.provenance:
            prov_cell.append(r.provenance, style=_click(th.s_60, "open_concept_crud", cname))
            prov_cell.append(" ", style=th.s_45)
        prov_cell.append("[edit]", style=_click(th.s_strong, "open_concept_crud", cname))
        prov_cell.append(" ", style=th.s_45)
        prov_cell.append("[×]", style=_click(th.s_45, "quick_delete_concept", cname))

        table.add_row(
            name,
            Text(r.category, style=_click(th.s_60, "open_concept_crud", cname)),
            Text(
                "" if r.confidence is None else f"conf {r.confidence:.2f}",
                style=_click(th.s_60, "open_concept_crud", cname),
            ),
            pin_cell,
            prov_cell,
        )
        if r.is_root and r.value and (r.pinned or data.search_query or data.as_of is not None):
            val_text = Text(
                f"   = {r.value}",
                style=_click(th.s_45, "open_concept_crud", cname),
                overflow="fold",
            )
            table.add_row(val_text, "", "", "", "")

    pinned = sum(1 for n in data.nodes if n.pinned)
    shown = sum(1 for r in rows if r.is_root)
    footer = Text(f"{pinned:,} pinned · {len(data.nodes):,} listed", style=th.s_45)
    if shown < len(data.nodes):
        footer.append(f" · showing {shown}, ", style=th.s_45)
        footer.append("scroll for list-all", style=_click(th.s_strong, "list_all"))
    footer.append("  ·  ", style=th.s_28)
    footer.append("[+ Create]", style=_click(th.s_strong, "create"))
    footer.append("  ", style=th.s_28)
    footer.append("[r Read]", style=_click(th.s_ink, "read"))
    footer.append("  ", style=th.s_28)
    footer.append("[u Update]", style=_click(th.s_ink, "update"))
    footer.append("  ", style=th.s_28)
    footer.append("[d Delete]", style=_click(th.s_ink, "delete"))

    graph_hint = Text()
    graph_hint.append("[+c Create]", style=_click(th.s_strong, "create"))
    graph_hint.append(" · ", style=th.s_45)
    graph_hint.append("[u Edit]", style=_click(th.s_ink, "update"))
    graph_hint.append(" · ", style=th.s_45)
    graph_hint.append("[d Del]", style=_click(th.s_ink, "delete"))
    graph_hint.append(" · 1-hop neighbourhood · pinned first", style=th.s_45)

    return Group(
        pane_heading("Graph", graph_hint, theme=th, label_action="open_crud_hub"),
        table,
        Text(""),
        footer,
    )


# ---------------------------------------------------------------------------
# AUDIT
# ---------------------------------------------------------------------------
def build_audit_pane(data: DashboardData, max_events: int = 3, max_chat_turns: int = 2) -> Group:
    th = get_theme(data.theme_mode)
    by_id = data.node_by_id
    audit_hint = Text()
    audit_hint.append("consensus · last capsule · ", style=th.s_45)
    audit_hint.append("[a] ask", style=_click(th.s_strong, "ask"))
    audit_hint.append(" · ", style=th.s_45)
    audit_hint.append("[p] inject", style=_click(th.s_ink, "pin_inject"))

    lines: List[RenderableType] = [pane_heading("Audit", audit_hint, theme=th, label_action="ask")]
    lines.append(Text("ConflictResolutionEvent", style=th.s_ink))
    events = sorted(data.conflicts, key=lambda e: -float(getattr(e, "resolved_at", 0.0)))[:max_events]
    if not events:
        lines.append(Text("  0 conflicts resolved", style=th.s_45))
    for ev in events:
        node = by_id.get(ev.node_id)
        cname = node.canonical_name if node else ev.node_id
        head = Text()
        head.append(_clip(ev.event_id, 18), style=_click(th.s_ink, "open_concept_crud", cname))
        head.append("  ")
        head.append(cname, style=_click(th.s_strong, "open_concept_crud", cname))
        head.append("  [inspect]", style=_click(th.s_45, "open_concept_crud", cname))
        lines.append(head)
        for label, agent, score, rev_id in (
            ("winner", ev.winning_agent_id, ev.winning_score, ev.winning_revision_id),
            ("loser ", ev.losing_agent_id, ev.losing_score, ev.losing_revision_id),
        ):
            t = Text("  ")
            t.append(label, style=th.s_45)
            t.append(f" {_clip(agent, 22):<22} ", style=_click(th.s_ink, "manage_agent", agent))
            t.append(f"{score:.3f}  ", style=th.s_ink)
            t.append(data.revision_provenance.get(rev_id, ""), style=th.s_60)
            lines.append(t)
        if ev.reason:
            t = Text("  ")
            t.append("reason", style=th.s_45)
            t.append(f" {ev.reason}", style=th.s_60)
            lines.append(t)
    lines.append(Text(""))

    cap = data.last_capsule
    if cap:
        target_aid = str(cap.get("target_agent_id", ""))
        head = Text("context_inject_capsule  ", style=_click(th.s_ink, "pin_inject"))
        head.append(target_aid, style=_click(th.s_strong, "manage_agent", target_aid))
        lines.append(head)
        sess = next((s for s in data.sessions if s.agent_id == target_aid), None)
        if sess is not None:
            t = Text(
                f"  window {sess.used_tokens:,} / {sess.max_context_tokens:,} ({sess.utilization_pct:.2f}%)  ",
                style=th.s_ink,
            )
            t.append_text(status_pill(sess.needs_compaction, theme=th, agent_id=target_aid))
            lines.append(t)
        lines.append(
            Text(
                f"  headroom {int(cap.get('remaining_headroom', 0)):,}   budget {int(cap.get('allocated_budget_tokens', 0)):,}"
                f"   capsule {int(cap.get('capsule_tokens', 0)):,}",
                style=th.s_ink,
            )
        )
        lines.append(
            Text(
                f"  raw {int(cap.get('raw_history_tokens_equivalent', 0)):,}   reduction {float(cap.get('token_reduction_pct', 0.0)):.2f}%",
                style=th.s_ink,
            )
        )
        if cap.get("sidecar_path"):
            lines.append(Text(f"  sidecar {cap['sidecar_path']}", style=th.s_60))
    else:
        t = Text("context_inject_capsule  ", style=th.s_ink)
        t.append(
            "no capsule pushed this session · [p] Pin & Inject",
            style=_click(th.s_45, "pin_inject"),
        )
        lines.append(t)
    lines.append(Text(""))

    # Conversational engine ([a] Ask): last turns, answered from the lake.
    if data.chat_history:
        for turn in data.chat_history[-max_chat_turns:]:
            you = Text("You:    ", style=th.s_45)
            you.append(str(turn.get("user", "")), style=_click(th.s_ink, "ask"))
            lines.append(you)
            answer_lines = str(turn.get("palace", "")).splitlines() or [""]
            for i, ans in enumerate(answer_lines[:5]):
                t = Text("Palace: " if i == 0 else "        ", style=th.s_45)
                t.append(ans, style=th.s_ink)
                lines.append(t)
            if len(answer_lines) > 5:
                lines.append(Text(f"        … {len(answer_lines) - 5} more", style=th.s_45))
            cited = turn.get("cited_nodes") or []
            if cited:
                c_line = Text("        └─ cited: ", style=th.s_60)
                for idx_c, c_node in enumerate(cited[:6]):
                    if idx_c:
                        c_line.append(", ", style=th.s_60)
                    c_str = str(c_node)
                    c_line.append(c_str, style=_click(th.s_strong, "open_concept_crud", c_str))
                lines.append(c_line)
        lines.append(Text(""))
    else:
        lines.append(
            Text(
                "[a] Ask the palace · e.g. 'What do my agents know about auth?', 'Remember port is 8443'",
                style=_click(th.s_45, "ask"),
            )
        )
        lines.append(Text(""))

    for line in data.last_output[-6:]:
        lines.append(Text(line, style=th.s_60, overflow="fold"))
    prompt = Text("palace$ ", style=_click(th.s_ink, "ask"))
    prompt.append(data.status_message, style=_click(th.s_60, "ask"))
    prompt.append(" █", style=_click(th.s_ink, "ask"))
    lines.append(prompt)
    return Group(*lines)


# ---------------------------------------------------------------------------
# full static render (context-palace tui --once)
# ---------------------------------------------------------------------------
def _pane(renderable: RenderableType, theme: Optional[ThemePalette] = None) -> Panel:
    th = theme or get_theme()
    return Panel(renderable, box=box.ROUNDED, border_style=th.s_ink, style=th.s_ink, padding=(0, 1))


def build_dashboard(data: DashboardData, width: int) -> Panel:
    th = get_theme(data.theme_mode)
    windows = _pane(build_windows_pane(data), theme=th)
    lake = _pane(build_lake_pane(data), theme=th)
    graph = _pane(build_graph_pane(data), theme=th)
    audit = _pane(build_audit_pane(data), theme=th)

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

    rule = Rule(style=th.s_28, characters="─")
    window = Group(build_header(data), Text(""), body, rule, build_status_bar(data, width=width))
    return Panel(window, box=box.ROUNDED, border_style=th.s_ink, style=th.s_ink, padding=(0, 1))
