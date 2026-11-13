"""
Interactive Textual front-end for the Context Palace v2 TUI.

Features:
- Dual-mode Dark ("Obsidian Mint" #0d1210 / #d8e6df) & Light ("Sage Mint" #e8efe6 / #1c1c1c) themes
  with live toggle via `[k] Theme` or clicking `[ ☾ DARK ]` / `[ ☀ LIGHT ]`.
- Mouse-clickable elements across all four panes (WINDOWS, LAKE, GRAPH, AUDIT), Header, and Status Bar.
- Full interactive CRUD modals (`ConceptCrudScreen`, `AgentCrudScreen`, `CrudHubScreen`) plus
  keyboard-driven `PromptScreen` for `[c] Create`, `[r] Read`, `[u] Update`, `[d] Delete`,
  `[l] List All`, `[a] Ask`, `[p] Pin & Inject`, `[m] Models`, `[t] Time-Travel`, `[k] Theme`, `[q] Quit`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rich.console import Group
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from context_palace.summarizer import (
    explain_concept_node,
    get_category_shape,
    summarize_3_to_5_words,
    summarize_concept,
)
from context_palace.tui_view import (
    BG,
    DARK_BG,
    DARK_INK,
    DARK_INK_16,
    DARK_INK_45,
    INK,
    INK_16,
    INK_45,
    LIGHT_BG,
    LIGHT_INK,
    LIGHT_INK_16,
    LIGHT_INK_45,
    STACK_BELOW_COLUMNS,
    ThemePalette,
    build_audit_pane,
    build_graph_pane,
    build_header,
    build_lake_pane,
    build_status_bar,
    build_windows_pane,
    get_theme,
)

Field = Tuple[str, str, str]  # (key, label, default)


def _modal_click(base_style: str, action: str, *args: Any) -> Style:
    meta = {"@click": (f"screen.{action}", tuple(args)) if args else f"screen.{action}"}
    return Style.parse(base_style) + Style(meta=meta)


class PromptScreen(ModalScreen[Optional[Dict[str, str]]]):
    """Hairline modal that collects the arguments for one hotkey action (with clickable buttons)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = f"""
    PromptScreen {{
        align: center middle;
        background: {DARK_BG} 70%;
    }}
    PromptScreen.theme-light {{
        background: {LIGHT_BG} 70%;
    }}
    PromptScreen > Vertical {{
        width: 76;
        height: auto;
        background: {DARK_BG};
        color: {DARK_INK};
        border: round {DARK_INK};
        padding: 1 2;
    }}
    PromptScreen.theme-light > Vertical {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border: round {LIGHT_INK};
    }}
    PromptScreen .prompt-title {{
        color: {DARK_INK};
        text-style: bold;
        margin-bottom: 1;
    }}
    PromptScreen.theme-light .prompt-title {{
        color: {LIGHT_INK};
    }}
    PromptScreen .prompt-label {{
        color: {DARK_INK_45};
    }}
    PromptScreen.theme-light .prompt-label {{
        color: {LIGHT_INK_45};
    }}
    PromptScreen .prompt-hint {{
        color: {DARK_INK_45};
        margin-top: 1;
    }}
    PromptScreen.theme-light .prompt-hint {{
        color: {LIGHT_INK_45};
    }}
    PromptScreen Input {{
        background: {DARK_BG};
        color: {DARK_INK};
        border: round {DARK_INK_45};
        padding: 0 1;
    }}
    PromptScreen.theme-light Input {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border: round {LIGHT_INK_45};
    }}
    PromptScreen Input:focus {{
        border: round {DARK_INK};
    }}
    PromptScreen.theme-light Input:focus {{
        border: round {LIGHT_INK};
    }}
    PromptScreen .prompt-actions {{
        height: auto;
        margin-top: 1;
    }}
    """

    def __init__(self, title: str, fields: Sequence[Field], hint: str = "", theme_mode: str = "dark") -> None:
        super().__init__()
        self.title_text = title
        self.fields = list(fields)
        self.hint = hint
        self.theme_mode = theme_mode

    def compose(self) -> ComposeResult:
        th = get_theme(self.theme_mode)
        with Vertical():
            yield Label(self.title_text, classes="prompt-title")
            for key, label, default in self.fields:
                yield Label(label.upper(), classes="prompt-label")
                yield Input(value=default, id=f"field-{key}")
            btn_line = Text()
            btn_line.append("[ Submit (Enter) ]", style=_modal_click(th.s_pill_on, "submit_now"))
            btn_line.append("   ")
            btn_line.append("[ Cancel (Esc) ]", style=_modal_click(th.s_ink, "cancel"))
            yield Static(btn_line, classes="prompt-actions")
            yield Label(
                (self.hint + "\n" if self.hint else "") + "enter = next / submit · esc or click = cancel",
                classes="prompt-hint",
            )

    def on_mount(self) -> None:
        self.set_class(self.theme_mode == "light", "theme-light")
        first = self.query(Input).first()
        if first:
            first.focus()

    def _collect(self) -> Dict[str, str]:
        return {key: self.query_one(f"#field-{key}", Input).value.strip() for key, _, _ in self.fields}

    def on_input_submitted(self, event: Input.Submitted) -> None:
        inputs = list(self.query(Input))
        idx = inputs.index(event.input)
        if idx < len(inputs) - 1:
            inputs[idx + 1].focus()
            return
        self.dismiss(self._collect())

    def action_submit_now(self) -> None:
        self.dismiss(self._collect())

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConceptCrudScreen(ModalScreen[Optional[Dict[str, Any]]]):
    """
    Full interactive CRUD Inspector & Editor for a ConceptNode (plus its BeliefRevisions and RelationEdges).
    Supports mouse clicks on every action button, revision row, and edge row.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+s", "save", "Save", show=False),
    ]

    DEFAULT_CSS = f"""
    ConceptCrudScreen {{
        align: center middle;
        background: {DARK_BG} 75%;
    }}
    ConceptCrudScreen.theme-light {{
        background: {LIGHT_BG} 75%;
    }}
    ConceptCrudScreen > VerticalScroll {{
        width: 96;
        max-height: 44;
        height: auto;
        background: {DARK_BG};
        color: {DARK_INK};
        border: round {DARK_INK};
        padding: 1 2;
    }}
    ConceptCrudScreen.theme-light > VerticalScroll {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border: round {LIGHT_INK};
    }}
    ConceptCrudScreen .crud-label {{
        color: {DARK_INK_45};
        margin-top: 0;
    }}
    ConceptCrudScreen.theme-light .crud-label {{
        color: {LIGHT_INK_45};
    }}
    ConceptCrudScreen Input {{
        background: {DARK_BG};
        color: {DARK_INK};
        border: round {DARK_INK_45};
        padding: 0 1;
    }}
    ConceptCrudScreen.theme-light Input {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border: round {LIGHT_INK_45};
    }}
    ConceptCrudScreen Input:focus {{
        border: round {DARK_INK};
    }}
    ConceptCrudScreen.theme-light Input:focus {{
        border: round {LIGHT_INK};
    }}
    """

    def __init__(
        self,
        tui: Any,
        node_id_or_name: str = "",
        is_create: bool = False,
        theme_mode: str = "dark",
    ) -> None:
        super().__init__()
        self.tui = tui
        self.target = node_id_or_name.strip()
        self.is_create = is_create or not self.target
        self.theme_mode = theme_mode
        self.node_obj: Any = None
        if not self.is_create and hasattr(self.tui.lake, "get_concept"):
            self.node_obj = self.tui.lake.get_concept(self.target, include_tombstoned=True)
            if self.node_obj is None:
                self.is_create = True
        self.pinned_state: bool = bool(getattr(self.node_obj, "pinned", False)) if self.node_obj else False

    def compose(self) -> ComposeResult:
        th = get_theme(self.theme_mode)
        name_val = getattr(self.node_obj, "canonical_name", self.target) if self.node_obj else self.target
        cat_val = getattr(self.node_obj, "category", "architecture") if self.node_obj else "architecture"
        val_val = getattr(self.node_obj, "active_value", "") if self.node_obj else ""
        conf_val = f"{float(getattr(self.node_obj, 'confidence', 0.95)):.2f}" if self.node_obj else "0.95"
        sum_val = getattr(self.node_obj, "summary", "") if self.node_obj else ""

        with VerticalScroll(id="crud-modal-scroll"):
            yield Static(id="crud-header-bar")
            yield Static(id="crud-explanation-box")
            yield Label(
                "1. UPDATE IN PLAIN ENGLISH (TERMINAL AGENT — NO FORM FILLING NEEDED)",
                classes="crud-label",
            )
            yield Input(
                value="",
                placeholder='e.g. "Change belief to support Homebrew + curl, set confidence to 99%, and link to context_palace_cli"',
                id="crud-nl-input",
            )
            yield Static(id="crud-nl-actions")
            yield Label(
                "2. AUTO-POPULATED FORM FIELDS (PRE-FILLED FROM SELECTED NODE — EDIT IF NEEDED)",
                classes="crud-label",
            )
            yield Label("CANONICAL NAME  (unique concept key in the memory lake)", classes="crud-label")
            yield Input(value=name_val, id="crud-name")
            yield Label("CATEGORY & SHAPE  (▲ architecture · ● config · ★ tool · ■ decision · ◆ general)", classes="crud-label")
            yield Input(value=cat_val, id="crud-category")
            yield Label("ACTIVE BELIEF VALUE  (what this concept currently states is true)", classes="crud-label")
            yield Input(value=val_val, id="crud-value")
            yield Label("CONFIDENCE SCORE (0.00 - 1.00)  &  SHORT TOPIC SUMMARY", classes="crud-label")
            yield Input(value=conf_val, id="crud-confidence")
            yield Input(value=sum_val, placeholder="Auto-generated 3-5 word summary if left blank", id="crud-summary")
            yield Label("ADD GRAPH CONNECTION  (target concept name  &  relationship type)", classes="crud-label")
            yield Input(value="", placeholder="Target concept name (e.g. ed25519_key_rotation)", id="crud-edge-target")
            yield Input(value="related_to", placeholder="Relation type (e.g. configures, depends_on)", id="crud-edge-rel")
            yield Static(id="crud-buttons-bar")
            yield Static(id="crud-relations-history")

    def on_mount(self) -> None:
        self.set_class(self.theme_mode == "light", "theme-light")
        self._refresh_chrome()
        if self.is_create:
            self.query_one("#crud-name", Input).focus()
        else:
            self.query_one("#crud-nl-input", Input).focus()

    def _refresh_chrome(self) -> None:
        from rich import box
        from rich.panel import Panel

        th = get_theme(self.theme_mode)
        cname = getattr(self.node_obj, "canonical_name", self.target) if self.node_obj else self.target
        cat = getattr(self.node_obj, "category", "architecture") if self.node_obj else "architecture"
        shape, _cat_lbl = get_category_shape(cat)
        mode_title = (
            "CREATE NEW CONCEPT NODE"
            if self.is_create
            else f"{shape} SELECTED NODE EXPLANATION & INSPECTOR · {cname}"
        )
        head = Table.grid(expand=True)
        head.add_column(justify="left")
        head.add_column(justify="right")
        pin_lbl = "★ PINNED: YES" if self.pinned_state else "☆ PINNED: NO"
        right = Text()
        right.append(f"[ {pin_lbl} ]", style=_modal_click(th.s_pill_on if self.pinned_state else th.s_ink, "toggle_pin"))
        right.append("  ")
        right.append("[ × Close (Esc) ]", style=_modal_click(th.s_ink, "cancel"))
        head.add_row(Text(mode_title, style=th.s_strong), right)
        self.query_one("#crud-header-bar", Static).update(head)

        # Build plain-English explanation box
        if not self.is_create and self.node_obj is not None:
            edges = (
                list(self.tui.lake.list_edges(node_id=self.node_obj.node_id, active_only=True))
                if hasattr(self.tui.lake, "list_edges")
                else []
            )
            revs = (
                list(self.tui.lake.get_belief_history(self.node_obj.node_id))
                if hasattr(self.tui.lake, "get_belief_history")
                else []
            )
            node_map = {
                n.node_id: n
                for n in (
                    self.tui.lake.list_concepts(include_tombstoned=True)
                    if hasattr(self.tui.lake, "list_concepts")
                    else []
                )
            }
            prov = revs[-1].provenance_type if revs else "agent_turn"
            expl = explain_concept_node(
                node=self.node_obj,
                provenance=prov,
                revision_count=max(1, len(revs)),
                edges=edges,
                node_by_id=node_map,
            )
            exp_lines = [
                Text.assemble(
                    (f"{expl['shape']} {expl['title']}  ", th.s_strong),
                    (f"[{self.node_obj.category.upper()} · {int(round(self.node_obj.confidence * 100))}% confidence]", th.s_60),
                ),
                Text.assemble(("• What this node is: ", th.s_strong), (expl["what_it_is"], th.s_ink)),
                Text.assemble(("• Why it matters:    ", th.s_strong), (expl["why_it_matters"], th.s_60)),
                Text.assemble(("• Recorded by:       ", th.s_strong), (expl["origin"], th.s_60)),
                Text.assemble(("• Graph links:       ", th.s_strong), (expl["connections"], th.s_60)),
            ]
            self.query_one("#crud-explanation-box", Static).update(
                Panel(
                    Group(*exp_lines),
                    title=Text(" WHAT THIS NODE REPRESENTS ", style=th.s_strong),
                    title_align="left",
                    border_style=th.s_45,
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
        else:
            self.query_one("#crud-explanation-box", Static).update(
                Text("Describe a new concept below in plain English or fill the form fields.", style=th.s_60)
            )

        nl_btns = Text()
        nl_btns.append(
            "[ ↵ Apply Plain-English Update ]",
            style=_modal_click(th.s_pill_on, "apply_nl"),
        )
        nl_btns.append(
            '   Examples: "set value to ...", "confidence 99%", "pin this", "link to X"',
            style=th.s_45,
        )
        self.query_one("#crud-nl-actions", Static).update(Group(nl_btns, Text("")))

        btns = Text()
        save_lbl = "[ + Create Concept ]" if self.is_create else "[ 💾 Save Form Changes ]"
        btns.append(save_lbl, style=_modal_click(th.s_pill_on, "save"))
        btns.append("  ")
        btns.append("[ 💉 Pin & Inject Capsule ]", style=_modal_click(th.s_strong, "pin_and_inject"))
        if not self.is_create:
            btns.append("  ")
            btns.append("[ 🗑 Soft Delete ]", style=_modal_click(th.s_ink, "soft_delete"))
            btns.append("  ")
            btns.append("[ ✕ Hard Delete ]", style=_modal_click(th.s_ink, "hard_delete"))
            btns.append("  ")
            btns.append("[ + New ]", style=_modal_click(th.s_ink, "switch_to_new"))
        btns.append("  ")
        btns.append("[ Cancel ]", style=_modal_click(th.s_45, "cancel"))
        self.query_one("#crud-buttons-bar", Static).update(Group(Text(""), btns, Text("")))

        # Render active edges & belief revision history with clickable actions
        details: List[Any] = []
        if not self.is_create and self.node_obj is not None:
            edges = (
                list(self.tui.lake.list_edges(node_id=self.node_obj.node_id, active_only=True))
                if hasattr(self.tui.lake, "list_edges")
                else []
            )
            details.append(Text(f"ACTIVE RELATION EDGES ({len(edges)})", style=th.s_label))
            if not edges:
                details.append(Text("  (no active edges · fill 'Add Relationship Edge' above and click Save)", style=th.s_45))
            for e in edges[:8]:
                row = Text("  ")
                row.append(f"{e.source_id} ", style=th.s_60)
                row.append(f"─[{e.relation_type}]→ ", style=th.s_ink)
                row.append(f"{e.target_id}  ", style=th.s_strong)
                row.append("[Delete Edge ×]", style=_modal_click(th.s_45, "remove_edge", e.edge_id))
                details.append(row)
            details.append(Text(""))

            revs = (
                list(self.tui.lake.get_belief_history(self.node_obj.node_id))
                if hasattr(self.tui.lake, "get_belief_history")
                else []
            )
            details.append(Text(f"BITEMPORAL BELIEF REVISION HISTORY ({len(revs)})", style=th.s_label))
            for rev in reversed(revs[-8:]):
                state = "ACTIVE" if rev.valid_to is None else "SUPERSEDED"
                rline = Text("  ")
                rline.append(f"{state:<10} ", style=th.s_strong if state == "ACTIVE" else th.s_45)
                rline.append(f"{rev.provenance_type:<14} ", style=th.s_60)
                rline.append(f"conf {rev.confidence:.2f}  ", style=th.s_60)
                rline.append(f"{rev.value[:42]}  ", style=th.s_ink)
                rline.append("[Restore Value]", style=_modal_click(th.s_strong, "restore_revision", rev.value))
                rline.append("  ")
                rline.append("[Time-Travel Here]", style=_modal_click(th.s_45, "time_travel_to", str(rev.valid_from)))
                details.append(rline)
        self.query_one("#crud-relations-history", Static).update(Group(*details) if details else Text(""))

    def _collect_fields(self) -> Dict[str, Any]:
        name = self.query_one("#crud-name", Input).value.strip()
        cat = self.query_one("#crud-category", Input).value.strip() or "architecture"
        val = self.query_one("#crud-value", Input).value.strip()
        conf_raw = self.query_one("#crud-confidence", Input).value.strip()
        summary = self.query_one("#crud-summary", Input).value.strip()
        edge_tgt = self.query_one("#crud-edge-target", Input).value.strip()
        edge_rel = self.query_one("#crud-edge-rel", Input).value.strip() or "related_to"
        try:
            conf = max(0.0, min(1.0, float(conf_raw)))
        except ValueError:
            conf = 0.95
        return {
            "canonical_name": name,
            "category": cat,
            "value": val,
            "confidence": conf,
            "summary": summary if summary else None,
            "pinned": self.pinned_state,
            "edge_target": edge_tgt,
            "edge_relation": edge_rel,
        }

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "crud-nl-input" and event.input.value.strip():
            self.action_apply_nl()
            return
        self.action_save()

    def action_apply_nl(self) -> None:
        nl_text = self.query_one("#crud-nl-input", Input).value.strip()
        if not nl_text:
            return
        default_name = getattr(self.node_obj, "canonical_name", self.target) if self.node_obj else self.target
        self.dismiss({"op": "nl", "instruction": nl_text, "default_node_name": default_name})

    def action_toggle_pin(self) -> None:
        self.pinned_state = not self.pinned_state
        self._refresh_chrome()

    def action_restore_revision(self, value: str) -> None:
        self.query_one("#crud-value", Input).value = value

    def action_time_travel_to(self, ts_str: str) -> None:
        try:
            ts = float(ts_str)
        except ValueError:
            return
        self.dismiss({"op": "time_travel", "as_of": ts})

    def action_remove_edge(self, edge_id: str) -> None:
        self.tui.execute_key_action("e", {"op": "delete", "edge_id": edge_id, "hard": False})
        self._refresh_chrome()

    def action_switch_to_new(self) -> None:
        self.is_create = True
        self.node_obj = None
        self.target = ""
        self.pinned_state = False
        self.query_one("#crud-name", Input).value = ""
        self.query_one("#crud-value", Input).value = ""
        self.query_one("#crud-summary", Input).value = ""
        self._refresh_chrome()
        self.query_one("#crud-name", Input).focus()

    def action_save(self) -> None:
        fields = self._collect_fields()
        if not fields["canonical_name"]:
            return
        if self.is_create:
            self.dismiss({"op": "create", "payload": fields})
        else:
            orig_id = getattr(self.node_obj, "node_id", self.target)
            fields["node_id_or_name"] = orig_id
            fields["new_value"] = fields["value"]
            fields["new_canonical_name"] = fields["canonical_name"]
            self.dismiss({"op": "update", "payload": fields})

    def action_pin_and_inject(self) -> None:
        fields = self._collect_fields()
        target_name = fields["canonical_name"] or self.target
        self.dismiss({"op": "pin_inject", "node_id_or_name": target_name})

    def action_soft_delete(self) -> None:
        orig_id = getattr(self.node_obj, "canonical_name", self.target)
        self.dismiss({"op": "delete", "node_id_or_name": orig_id, "hard": False})

    def action_hard_delete(self) -> None:
        orig_id = getattr(self.node_obj, "canonical_name", self.target)
        self.dismiss({"op": "delete", "node_id_or_name": orig_id, "hard": True})

    def action_cancel(self) -> None:
        self.dismiss(None)


class AgentCrudScreen(ModalScreen[Optional[Dict[str, Any]]]):
    """Interactive CRUD Modal for creating, updating, compacting, or deleting an AgentSession."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = f"""
    AgentCrudScreen {{
        align: center middle;
        background: {DARK_BG} 75%;
    }}
    AgentCrudScreen.theme-light {{
        background: {LIGHT_BG} 75%;
    }}
    AgentCrudScreen > Vertical {{
        width: 82;
        height: auto;
        background: {DARK_BG};
        color: {DARK_INK};
        border: round {DARK_INK};
        padding: 1 2;
    }}
    AgentCrudScreen.theme-light > Vertical {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border: round {LIGHT_INK};
    }}
    AgentCrudScreen .crud-label {{
        color: {DARK_INK_45};
    }}
    AgentCrudScreen.theme-light .crud-label {{
        color: {LIGHT_INK_45};
    }}
    AgentCrudScreen Input {{
        background: {DARK_BG};
        color: {DARK_INK};
        border: round {DARK_INK_45};
        padding: 0 1;
    }}
    AgentCrudScreen.theme-light Input {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border: round {LIGHT_INK_45};
    }}
    """

    def __init__(self, tui: Any, agent_id: str = "", theme_mode: str = "dark") -> None:
        super().__init__()
        self.tui = tui
        self.agent_id = agent_id.strip()
        self.theme_mode = theme_mode
        self.sess = (
            self.tui.lake.get_agent_session(self.agent_id)
            if (self.agent_id and hasattr(self.tui.lake, "get_agent_session"))
            else None
        )

    def compose(self) -> ComposeResult:
        from rich import box
        from rich.panel import Panel

        th = get_theme(self.theme_mode)
        aid = self.sess.agent_id if self.sess else self.agent_id
        harness = self.sess.harness_type if self.sess else "antigravity"
        model = self.sess.model_id if self.sess else "gemini-4-argon"
        used = str(self.sess.used_tokens) if self.sess else "0"
        max_tok = str(self.sess.max_context_tokens) if self.sess else "1000000"

        title = f"AGENT SESSION INSPECTOR & CRUD · {aid}" if aid else "REGISTER NEW AGENT SESSION"
        with Vertical():
            yield Label(title, classes="prompt-title")
            if self.sess is not None:
                nodes = (
                    list(self.tui.lake.list_concepts(include_tombstoned=False))
                    if hasattr(self.tui.lake, "list_concepts")
                    else []
                )
                owned = [n for n in nodes if n.source_agent_id == aid]
                owned.sort(key=lambda n: n.updated_at, reverse=True)
                latest_text = owned[0].active_value if owned else f"{harness} {model} session"
                topic_3_5 = summarize_3_to_5_words(latest_text, fallback=aid)
                headroom = max(0, self.sess.max_context_tokens - self.sess.used_tokens)
                status_explain = (
                    "HEALTHY — plenty of context window headroom remaining."
                    if not self.sess.needs_compaction
                    else "NEEDS COMPACTION — over 75% of context window used; click Compact below."
                )
                sess_lines = [
                    Text.assemble(("• Latest Topic (3–5w): ", th.s_strong), (topic_3_5, th.s_ink)),
                    Text.assemble(
                        ("• Latest Activity:     ", th.s_strong),
                        (latest_text[:72], th.s_60),
                    ),
                    Text.assemble(
                        ("• Window Health:       ", th.s_strong),
                        (
                            f"{self.sess.utilization_pct:.2f}% used ({headroom:,} tokens free) — {status_explain}",
                            th.s_60,
                        ),
                    ),
                    Text.assemble(
                        ("• Concepts Recorded:   ", th.s_strong),
                        (f"{len(owned)} active concept node(s) in the lake", th.s_60),
                    ),
                ]
                yield Static(
                    Panel(
                        Group(*sess_lines),
                        title=Text(" WHAT THIS SESSION REPRESENTS ", style=th.s_strong),
                        title_align="left",
                        border_style=th.s_45,
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
            yield Label("AUTO-POPULATED SESSION FIELDS (EDIT BELOW IF NEEDED)", classes="crud-label")
            yield Label("AGENT ID", classes="crud-label")
            yield Input(value=aid, id="agent-id")
            yield Label("HARNESS TYPE (antigravity · claude-code · codex · ollama)", classes="crud-label")
            yield Input(value=harness, id="agent-harness")
            yield Label("MODEL ID", classes="crud-label")
            yield Input(value=model, id="agent-model")
            yield Label("USED TOKENS", classes="crud-label")
            yield Input(value=used, id="agent-used")
            yield Label("MAX CONTEXT WINDOW TOKENS", classes="crud-label")
            yield Input(value=max_tok, id="agent-max")

            btns = Text()
            btns.append("[ 💾 Save Session ]", style=_modal_click(th.s_pill_on, "save"))
            btns.append("  ")
            btns.append("[ 💉 Compact / Inject Capsule ]", style=_modal_click(th.s_strong, "compact"))
            if self.sess is not None:
                btns.append("  ")
                btns.append("[ 🗑 Delete Session ]", style=_modal_click(th.s_ink, "delete_session"))
            btns.append("  ")
            btns.append("[ Cancel (Esc) ]", style=_modal_click(th.s_45, "cancel"))
            yield Static(Group(Text(""), btns))

    def on_mount(self) -> None:
        self.set_class(self.theme_mode == "light", "theme-light")
        self.query_one("#agent-id", Input).focus()

    def _collect(self) -> Dict[str, Any]:
        aid = self.query_one("#agent-id", Input).value.strip()
        harness = self.query_one("#agent-harness", Input).value.strip() or "antigravity"
        model = self.query_one("#agent-model", Input).value.strip() or "gemini-4-argon"
        try:
            used = max(0, int(self.query_one("#agent-used", Input).value.replace(",", "").strip()))
        except ValueError:
            used = 0
        try:
            max_tok = max(1024, int(self.query_one("#agent-max", Input).value.replace(",", "").strip()))
        except ValueError:
            max_tok = 1_000_000
        return {
            "agent_id": aid,
            "harness_type": harness,
            "model_id": model,
            "used_tokens": used,
            "max_context_tokens": max_tok,
        }

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_save()

    def action_save(self) -> None:
        data = self._collect()
        if not data["agent_id"]:
            return
        self.dismiss({"op": "upsert", "payload": data})

    def action_compact(self) -> None:
        data = self._collect()
        if not data["agent_id"]:
            return
        self.dismiss({"op": "compact", "payload": data})

    def action_delete_session(self) -> None:
        aid = self.query_one("#agent-id", Input).value.strip() or self.agent_id
        self.dismiss({"op": "delete", "agent_id": aid})

    def action_cancel(self) -> None:
        self.dismiss(None)


class CrudHubScreen(ModalScreen[Optional[Dict[str, Any]]]):
    """
    Full-screen clickable CRUD Manager Hub listing all Concepts (active + tombstoned) and Agent Sessions
    with direct one-click [Edit], [Pin/Unpin], [Restore], [Soft Delete], [Hard Delete], and [+ Create] actions.
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
        Binding("c", "new_concept", "New Concept", show=False),
    ]

    DEFAULT_CSS = f"""
    CrudHubScreen {{
        background: {DARK_BG};
        padding: 1 2;
    }}
    CrudHubScreen.theme-light {{
        background: {LIGHT_BG};
    }}
    CrudHubScreen > VerticalScroll {{
        background: {DARK_BG};
        color: {DARK_INK};
        border: round {DARK_INK};
        padding: 1 2;
    }}
    CrudHubScreen.theme-light > VerticalScroll {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border: round {LIGHT_INK};
    }}
    """

    def __init__(self, tui: Any, theme_mode: str = "dark") -> None:
        super().__init__()
        self.tui = tui
        self.theme_mode = theme_mode

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="crud-hub-scroll"):
            yield Static(id="crud-hub-body")

    def on_mount(self) -> None:
        self.set_class(self.theme_mode == "light", "theme-light")
        self._render_hub()

    def _render_hub(self) -> None:
        th = get_theme(self.theme_mode)
        nodes = (
            list(self.tui.lake.list_concepts(include_tombstoned=True))
            if hasattr(self.tui.lake, "list_concepts")
            else []
        )
        sessions = self.tui._fetch_agent_sessions()

        top = Table.grid(expand=True)
        top.add_column(justify="left")
        top.add_column(justify="right")
        actions = Text()
        actions.append("[ + Create Concept ]", style=_modal_click(th.s_pill_on, "new_concept"))
        actions.append("  ")
        actions.append("[ + Register Agent ]", style=_modal_click(th.s_strong, "new_agent"))
        actions.append("  ")
        actions.append("[ × Back to Palace (Esc) ]", style=_modal_click(th.s_ink, "close"))
        top.add_row(
            Text(f"CRUD MANAGER HUB  ·  {len(nodes)} concepts · {len(sessions)} agent sessions", style=th.s_strong),
            actions,
        )

        ctBL = Table(
            show_header=True,
            show_lines=True,
            show_edge=False,
            expand=True,
            border_style=th.s_28,
            header_style=th.s_label,
            style=th.s_ink,
        )
        ctBL.add_column("CONCEPT", ratio=3)
        ctBL.add_column("CATEGORY", ratio=2)
        ctBL.add_column("ACTIVE BELIEF VALUE", ratio=4)
        ctBL.add_column("CONF", justify="right")
        ctBL.add_column("STATE", justify="center")
        ctBL.add_column("CRUD ACTIONS", justify="right", ratio=3)

        for n in nodes:
            cname = n.canonical_name
            state_txt = "TOMBSTONED" if n.tombstoned else ("PINNED" if n.pinned else "ACTIVE")
            act = Text()
            act.append("[Edit]", style=_modal_click(th.s_strong, "edit_concept", cname))
            act.append(" ")
            act.append(
                "[Unpin]" if n.pinned else "[Pin]",
                style=_modal_click(th.s_ink, "toggle_pin", cname),
            )
            act.append(" ")
            if n.tombstoned:
                act.append("[Restore]", style=_modal_click(th.s_strong, "restore_concept", cname))
            else:
                act.append("[Soft Del]", style=_modal_click(th.s_45, "soft_del", cname))
            act.append(" ")
            act.append("[Hard Del ×]", style=_modal_click(th.s_45, "hard_del", cname))

            ctBL.add_row(
                Text(cname, style=_modal_click(th.s_strong, "edit_concept", cname)),
                Text(n.category, style=th.s_60),
                Text(n.active_value[:56], style=_modal_click(th.s_ink, "edit_concept", cname)),
                Text(f"{n.confidence:.2f}", style=th.s_60),
                Text(state_txt, style=th.s_pill_on if n.pinned and not n.tombstoned else th.s_60),
                act,
            )

        stBL = Table(
            show_header=True,
            show_lines=True,
            show_edge=False,
            expand=True,
            border_style=th.s_28,
            header_style=th.s_label,
            style=th.s_ink,
        )
        stBL.add_column("AGENT ID", ratio=3)
        stBL.add_column("HARNESS", ratio=2)
        stBL.add_column("MODEL", ratio=3)
        stBL.add_column("UTILIZATION", ratio=3)
        stBL.add_column("CRUD ACTIONS", justify="right", ratio=3)
        for s in sessions:
            aid = s.agent_id
            sact = Text()
            sact.append("[Edit]", style=_modal_click(th.s_strong, "edit_agent", aid))
            sact.append(" ")
            sact.append("[Compact]", style=_modal_click(th.s_ink, "compact_agent", aid))
            sact.append(" ")
            sact.append("[Delete ×]", style=_modal_click(th.s_45, "del_agent", aid))
            stBL.add_row(
                Text(aid, style=_modal_click(th.s_strong, "edit_agent", aid)),
                Text(s.harness_type, style=th.s_60),
                Text(s.model_id, style=th.s_60),
                Text(f"{s.used_tokens:,} / {s.max_context_tokens:,} ({s.utilization_pct:.1f}%)", style=th.s_ink),
                sact,
            )

        self.query_one("#crud-hub-body", Static).update(
            Group(
                top,
                Text(""),
                Text("CONCEPT NODES (CLICK ANY ROW OR ACTION TO MANAGE)", style=th.s_label),
                ctBL,
                Text(""),
                Text("AGENT SESSIONS", style=th.s_label),
                stBL,
            )
        )

    def action_new_concept(self) -> None:
        self.dismiss({"next": "create_concept"})

    def action_new_agent(self) -> None:
        self.dismiss({"next": "edit_agent", "agent_id": ""})

    def action_edit_concept(self, name: str) -> None:
        self.dismiss({"next": "edit_concept", "name": name})

    def action_toggle_pin(self, name: str) -> None:
        node = self.tui.lake.get_concept(name, include_tombstoned=True)
        if node is not None:
            self.tui.lake.pin_concept(name, pinned=not node.pinned)
        self._render_hub()

    def action_restore_concept(self, name: str) -> None:
        node = self.tui.lake.get_concept(name, include_tombstoned=True)
        if node is not None:
            self.tui.execute_key_action("u", {"node_id_or_name": name, "new_value": node.active_value})
        self._render_hub()

    def action_soft_del(self, name: str) -> None:
        self.tui.execute_key_action("d", {"node_id_or_name": name, "hard": False})
        self._render_hub()

    def action_hard_del(self, name: str) -> None:
        self.tui.execute_key_action("d", {"node_id_or_name": name, "hard": True})
        self._render_hub()

    def action_edit_agent(self, agent_id: str) -> None:
        self.dismiss({"next": "edit_agent", "agent_id": agent_id})

    def action_compact_agent(self, agent_id: str) -> None:
        self.tui.execute_key_action("p", {"agent_id": agent_id, "query": "compaction"})
        self._render_hub()

    def action_del_agent(self, agent_id: str) -> None:
        self.tui.execute_key_action("s", {"op": "delete", "agent_id": agent_id})
        self._render_hub()

    def action_close(self) -> None:
        self.dismiss(None)


class InventoryScreen(ModalScreen[None]):
    """Full-screen, scrollable, untruncated lake inventory ([l] List All)."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
        Binding("l", "close", "Close", show=False),
    ]

    DEFAULT_CSS = f"""
    InventoryScreen {{
        background: {DARK_BG};
        padding: 1 2;
    }}
    InventoryScreen.theme-light {{
        background: {LIGHT_BG};
    }}
    InventoryScreen > VerticalScroll {{
        background: {DARK_BG};
        color: {DARK_INK};
        border: round {DARK_INK};
        padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-color: {DARK_INK_45};
        scrollbar-color-hover: {DARK_INK};
        scrollbar-color-active: {DARK_INK};
        scrollbar-background: {DARK_BG};
        scrollbar-background-hover: {DARK_BG};
        scrollbar-background-active: {DARK_BG};
    }}
    InventoryScreen.theme-light > VerticalScroll {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border: round {LIGHT_INK};
        scrollbar-color: {LIGHT_INK_45};
        scrollbar-color-hover: {LIGHT_INK};
        scrollbar-color-active: {LIGHT_INK};
        scrollbar-background: {LIGHT_BG};
        scrollbar-background-hover: {LIGHT_BG};
        scrollbar-background-active: {LIGHT_BG};
    }}
    InventoryScreen Static {{
        background: {DARK_BG};
        color: {DARK_INK};
    }}
    InventoryScreen.theme-light Static {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
    }}
    InventoryScreen .inv-title {{
        text-style: bold;
    }}
    InventoryScreen .inv-hint {{
        color: {DARK_INK_45};
    }}
    InventoryScreen.theme-light .inv-hint {{
        color: {LIGHT_INK_45};
    }}
    """

    def __init__(self, title: str, body: str, theme_mode: str = "dark") -> None:
        super().__init__()
        self.title_text = title
        self.body = body
        self.theme_mode = theme_mode

    def compose(self) -> ComposeResult:
        th = get_theme(self.theme_mode)
        with VerticalScroll(id="inventory"):
            yield Static("LIST ALL  ·  " + self.title_text, classes="inv-title")
            hint = Text("esc / q / l = back to the palace  ·  ")
            hint.append("[ × Close Inventory ]", style=_modal_click(th.s_strong, "close"))
            yield Static(hint, classes="inv-hint")
            yield Static(Text(self.body), id="inventory-body")

    def on_mount(self) -> None:
        self.set_class(self.theme_mode == "light", "theme-light")

    def action_close(self) -> None:
        self.dismiss(None)


class DragDropCard(Static):
    """Draggable / Droppable card widget inside DragDropStudioScreen."""

    allow_select = False

    def __init__(
        self,
        renderable: Any = "",
        dnd_kind: str = "",
        dnd_id: str = "",
        dnd_label: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(renderable, **kwargs)
        self.dnd_kind = dnd_kind  # "concept", "window", "lake_dropzone", or "static"
        self.dnd_id = dnd_id
        self.dnd_label = dnd_label

    def set_card(
        self,
        renderable: Any,
        dnd_kind: str = "",
        dnd_id: str = "",
        dnd_label: str = "",
    ) -> None:
        self.dnd_kind = dnd_kind
        self.dnd_id = dnd_id
        self.dnd_label = dnd_label
        self.display = True
        self.update(renderable)


class DragDropStudioScreen(ModalScreen[None]):
    """
    Split-screen Drag & Drop Context Studio bridging:
      Left Column:  Centralized SQLite Context Lake (~/.context_palace/context_lake.db)
      Right Column: Actual Open Agent Context Windows (Live running terminal sessions only)
    Supports both continuous mouse drag-and-drop (press -> move -> release) and click-to-transfer,
    with 3-5 word summaries on every card.
    """

    allow_select = False

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("w", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
        Binding("m", "sync_windows", "Sync Windows", show=False),
    ]

    DEFAULT_CSS = f"""
    DragDropStudioScreen {{
        background: {DARK_BG};
        padding: 1 2;
    }}
    DragDropStudioScreen.theme-light {{
        background: {LIGHT_BG};
    }}
    #dnd-root {{
        background: {DARK_BG};
        color: {DARK_INK};
        border: round {DARK_INK};
        padding: 1 2;
        height: 1fr;
    }}
    DragDropStudioScreen.theme-light #dnd-root {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border: round {LIGHT_INK};
    }}
    #dnd-header {{
        height: auto;
        margin-bottom: 1;
    }}
    #dnd-status-banner {{
        height: auto;
        margin-bottom: 1;
    }}
    #dnd-columns {{
        height: 1fr;
    }}
    .dnd-col {{
        width: 1fr;
        height: 1fr;
        border: round {DARK_INK_45};
        padding: 0 1;
        margin-right: 1;
    }}
    DragDropStudioScreen.theme-light .dnd-col {{
        border: round {LIGHT_INK_45};
    }}
    .dnd-card {{
        height: auto;
        margin-bottom: 1;
    }}
    """

    def __init__(self, tui: Any, theme_mode: str = "dark", initial_drag: Optional[Dict[str, str]] = None) -> None:
        super().__init__()
        self.tui = tui
        self.theme_mode = theme_mode
        self.drag_state: Optional[Dict[str, str]] = initial_drag

    def compose(self) -> ComposeResult:
        with Vertical(id="dnd-root"):
            yield Static(id="dnd-header")
            yield Static(id="dnd-status-banner")
            with Horizontal(id="dnd-columns"):
                with VerticalScroll(classes="dnd-col", id="dnd-lake-col"):
                    yield Vertical(id="dnd-lake-cards")
                with VerticalScroll(classes="dnd-col", id="dnd-windows-col"):
                    yield Vertical(id="dnd-windows-cards")

    def on_mount(self) -> None:
        self.set_class(self.theme_mode == "light", "theme-light")
        self._render_studio()

    def _update_banner_only(self) -> None:
        from rich import box
        from rich.panel import Panel

        th = get_theme(self.theme_mode)
        if self.drag_state:
            kind = self.drag_state.get("kind", "concept")
            lbl = self.drag_state.get("label", self.drag_state.get("id", ""))
            src_id = self.drag_state.get("id", "")
            target_hint = (
                "Release mouse over (or click) any Open Context Window on the RIGHT to inject this SQLite context!"
                if kind == "concept"
                else "Release mouse over (or click) the Centralized SQLite Lake on the LEFT (or another Window) to drop!"
            )
            txt = Text()
            txt.append(f"✋ DRAGGING {kind.upper()}: '{lbl}' ({src_id})  ──►  ", style=th.s_strong)
            txt.append(target_hint, style=th.s_ink)
            txt.append("   ", style=th.s_45)
            txt.append("[ × Cancel Drag ]", style=_modal_click(th.s_45, "cancel_drag"))
        else:
            txt = Text(
                "Drag & Drop any card between the CENTRALIZED SQLITE LAKE (Left) and ACTUAL OPEN AGENT WINDOWS (Right) — or click a card to pick it up and click destination to drop.",
                style=th.s_60,
            )
        self.query_one("#dnd-status-banner", Static).update(
            Panel(txt, box=box.ROUNDED, border_style=th.s_strong if self.drag_state else th.s_28, padding=(0, 1))
        )

    def _sync_card_list(
        self,
        container: Vertical,
        specs: List[Tuple[Any, str, str, str]],
    ) -> None:
        """
        Update existing DragDropCard children in-place without ever calling .remove().
        This guarantees no detached widget with `parent is None` can ever be hit by Textual's MouseDown dispatcher.
        """
        existing = [c for c in container.children if isinstance(c, DragDropCard)]
        for i, (renderable, kind, cid, label) in enumerate(specs):
            if i < len(existing):
                existing[i].set_card(renderable, dnd_kind=kind, dnd_id=cid, dnd_label=label)
            else:
                container.mount(
                    DragDropCard(
                        renderable,
                        dnd_kind=kind,
                        dnd_id=cid,
                        dnd_label=label,
                        classes="dnd-card",
                    )
                )
        for j in range(len(specs), len(existing)):
            existing[j].dnd_kind = ""
            existing[j].dnd_id = ""
            existing[j].dnd_label = ""
            existing[j].display = False

    def _render_studio(self) -> None:
        from rich import box
        from rich.panel import Panel
        from context_palace.summarizer import get_session_topic_summary

        th = get_theme(self.theme_mode)
        sessions = self.tui._fetch_agent_sessions()
        nodes = (
            list(self.tui.lake.list_concepts(include_tombstoned=False))
            if hasattr(self.tui.lake, "list_concepts")
            else []
        )

        # Header
        top = Table.grid(expand=True)
        top.add_column(justify="left")
        top.add_column(justify="right")
        actions = Text()
        actions.append("[ ⟳ Sync Live Open Windows (m) ]", style=_modal_click(th.s_strong, "sync_windows"))
        actions.append("  ")
        actions.append("[ × Back to Palace (Esc / w) ]", style=_modal_click(th.s_ink, "close"))
        top.add_row(
            Text(
                f"⇄ DRAG & DROP CONTEXT STUDIO  ·  {len(nodes)} SQLite Lake Concepts ↔ {len(sessions)} Live Open Window(s)",
                style=th.s_strong,
            ),
            actions,
        )
        self.query_one("#dnd-header", Static).update(top)
        self._update_banner_only()

        # Map which open windows currently have each concept injected
        concept_to_windows: Dict[str, List[str]] = {}
        for s in sessions:
            meta = s.metadata if isinstance(getattr(s, "metadata", None), dict) else {}
            for cname in (meta.get("injected_concepts") or []):
                concept_to_windows.setdefault(cname, []).append(s.agent_id)

        # Build Left Column Specs: Centralized SQLite Lake
        lake_container = self.query_one("#dnd-lake-cards", Vertical)
        lake_specs: List[Tuple[Any, str, str, str]] = []

        lake_drop_txt = Text()
        lake_drop_txt.append(
            "1. CENTRALIZED CONTEXT LAKE (SQLite DB: ~/.context_palace/context_lake.db)\n",
            style=th.s_strong,
        )
        lake_drop_txt.append(
            "[ ⬇ DROP LIVE WINDOW CONTEXT HERE TO SAVE INTO CENTRALIZED SQLITE DB ]",
            style=_modal_click(th.s_pill_on if (self.drag_state and self.drag_state.get("kind") == "window") else th.s_strong, "drop_on_lake"),
        )
        lake_specs.append(
            (
                Panel(lake_drop_txt, box=box.ROUNDED, border_style=th.s_45, padding=(0, 1)),
                "lake_dropzone",
                "sqlite_lake",
                "Centralized SQLite Lake",
            )
        )

        for idx, n in enumerate(nodes, start=1):
            cname = n.canonical_name
            shape, _ = get_category_shape(n.category)
            c_info = summarize_concept(cname, n.active_value, getattr(n, "summary", ""), n.category)
            short_topic = c_info["short_topic"]
            plain = c_info["plain_english"]
            in_wins = concept_to_windows.get(cname, [])

            l1 = Text()
            l1.append("[≡ DRAG] ", style=_modal_click(th.s_pill_on, "pickup_concept", cname))
            l1.append(f"{shape} #{idx} · {short_topic} ", style=_modal_click(th.s_strong, "pickup_concept", cname))
            l1.append(f"({cname} · {int(round(n.confidence * 100))}%)", style=th.s_60)
            if n.pinned:
                l1.append(" ★ PINNED", style=th.s_strong)

            l2 = Text(f"  Summary: {plain}", style=th.s_ink, overflow="fold")

            l3 = Text("  ")
            if in_wins:
                l3.append(f"In Windows: {', '.join(in_wins)}   ", style=th.s_strong)
            for s in sessions:
                l3.append(
                    f"[ ➔ Drop to {s.agent_id} ] ",
                    style=_modal_click(th.s_strong, "quick_drop_to_window", cname, s.agent_id),
                )

            card_panel = Panel(
                Group(l1, l2, l3),
                box=box.ROUNDED,
                border_style=th.s_strong if (self.drag_state and self.drag_state.get("id") == cname) else th.s_28,
                padding=(0, 1),
            )
            lake_specs.append((card_panel, "concept", cname, short_topic))

        self._sync_card_list(lake_container, lake_specs)

        # Build Right Column Specs: Actual Open Context Windows
        win_container = self.query_one("#dnd-windows-cards", Vertical)
        win_specs: List[Tuple[Any, str, str, str]] = []

        win_hdr = Text(
            f"2. ACTUAL OPEN CONTEXT WINDOWS ({len(sessions)} Live Running Terminal Sessions · Closed Windows Pruned)",
            style=th.s_strong,
        )
        win_specs.append((win_hdr, "static", "windows_header", "Open Windows Header"))

        if not sessions:
            win_specs.append(
                (
                    Panel(
                        Text("(0 open agent sessions detected — launch `agy` or `claude` in a terminal)"),
                        box=box.ROUNDED,
                        border_style=th.s_28,
                    ),
                    "static",
                    "windows_empty",
                    "Empty Windows",
                )
            )
        node_by_cname = {n.canonical_name: n for n in nodes}
        for s in sessions:
            aid = s.agent_id
            topic_3_5w, latest_detail = get_session_topic_summary(s)
            meta = s.metadata if isinstance(getattr(s, "metadata", None), dict) else {}
            injected = list(meta.get("injected_concepts") or [])

            w1 = Text()
            w1.append(f"● OPEN WINDOW: {aid}  ", style=th.s_strong)
            w1.append(f"({s.harness_type} · {s.model_id} · {s.used_tokens:,}/{s.max_context_tokens:,} tok · {s.utilization_pct:.1f}%)", style=th.s_60)

            w2 = Text.assemble(
                ("  Window Topic Summary (3-5w): ", th.s_60),
                (topic_3_5w, th.s_strong),
            )
            w3 = Text(f"  Latest Prompt: \"{latest_detail[:82]}\"", style=th.s_60, overflow="fold")

            w_actions = Text("  ")
            w_actions.append(
                f"[ ⬇ DROP SQLITE CONTEXT INTO {aid} ]",
                style=_modal_click(
                    th.s_pill_on if (self.drag_state and self.drag_state.get("kind") == "concept") else th.s_strong,
                    "drop_on_window",
                    aid,
                ),
            )
            w_actions.append("   ")
            w_actions.append(
                "[ ⬅ Drag / Save Window Topic to SQLite Lake ]",
                style=_modal_click(th.s_ink, "save_window_to_lake", aid),
            )

            inj_lines: List[Any] = [
                Text(f"  SQLite Context Items Injected in {aid} ({len(injected)}):", style=th.s_label)
            ]
            if not injected:
                inj_lines.append(
                    Text("    (Drag any concept card from the Left SQLite Lake and drop it here)", style=th.s_45)
                )
            else:
                for ic_name in injected:
                    ic_node = node_by_cname.get(ic_name)
                    if ic_node:
                        ic_shape, _ = get_category_shape(ic_node.category)
                        ic_info = summarize_concept(ic_name, ic_node.active_value, ic_node.summary, ic_node.category)
                        ic_summary = f"{ic_shape} {ic_info['short_topic']} — {ic_info['plain_english'][:52]}"
                    else:
                        ic_summary = f"◆ {ic_name}"
                    row_t = Text("    • ")
                    row_t.append(ic_summary, style=th.s_ink)
                    row_t.append(f" ({ic_name})  ", style=th.s_60)
                    row_t.append("[× Remove]", style=_modal_click(th.s_45, "remove_from_window", ic_name, aid))
                    inj_lines.append(row_t)

            w_panel = Panel(
                Group(w1, w2, w3, Text(""), w_actions, Text(""), *inj_lines),
                box=box.ROUNDED,
                border_style=th.s_strong if (self.drag_state and self.drag_state.get("kind") == "concept") else th.s_45,
                padding=(0, 1),
            )
            win_specs.append((w_panel, "window", aid, topic_3_5w))

        self._sync_card_list(win_container, win_specs)

    def _find_dnd_card_at(self, screen_x: int, screen_y: int) -> Optional[DragDropCard]:
        try:
            widget, _ = self.get_widget_at(screen_x, screen_y)
        except Exception:
            return None
        cur: Any = widget
        while cur is not None:
            if isinstance(cur, DragDropCard):
                return cur
            cur = getattr(cur, "parent", None)
        return None

    def on_mouse_down(self, event: Any) -> None:
        card = self._find_dnd_card_at(event.screen_x, event.screen_y)
        if card is not None and card.dnd_kind in ("concept", "window"):
            self._mouse_down_card = card
            self._mouse_down_pos = (event.screen_x, event.screen_y)

    def on_mouse_move(self, event: Any) -> None:
        down_card = getattr(self, "_mouse_down_card", None)
        down_pos = getattr(self, "_mouse_down_pos", None)
        if down_card is not None and down_pos is not None:
            if abs(event.screen_x - down_pos[0]) + abs(event.screen_y - down_pos[1]) >= 2:
                if not self.drag_state or self.drag_state.get("id") != down_card.dnd_id:
                    self.drag_state = {
                        "kind": down_card.dnd_kind,
                        "id": down_card.dnd_id,
                        "label": down_card.dnd_label,
                    }
                    self._update_banner_only()

    def on_mouse_up(self, event: Any) -> None:
        down_card = getattr(self, "_mouse_down_card", None)
        self._mouse_down_card = None
        self._mouse_down_pos = None
        if not self.drag_state:
            return
        target_card = self._find_dnd_card_at(event.screen_x, event.screen_y)
        if target_card is None or (down_card is not None and target_card is down_card):
            return
        src_kind = self.drag_state.get("kind")
        src_id = self.drag_state.get("id", "")
        if src_kind == "concept" and target_card.dnd_kind == "window":
            self.drag_state = None
            self.tui.execute_key_action("dnd", {"op": "lake_to_window", "node_id_or_name": src_id, "agent_id": target_card.dnd_id})
            self._render_studio()
        elif src_kind == "window" and target_card.dnd_kind in ("lake_dropzone", "concept"):
            self.drag_state = None
            self.tui.execute_key_action("dnd", {"op": "window_to_lake", "agent_id": src_id})
            self._render_studio()
        elif src_kind == "window" and target_card.dnd_kind == "window" and target_card.dnd_id != src_id:
            self.drag_state = None
            self.tui.execute_key_action("dnd", {"op": "window_to_window", "source_agent_id": src_id, "target_agent_id": target_card.dnd_id})
            self._render_studio()

    def action_pickup_concept(self, cname: str) -> None:
        node = self.tui.lake.get_concept(cname, include_tombstoned=True)
        c_info = summarize_concept(cname, node.active_value if node else "")
        self.drag_state = {"kind": "concept", "id": cname, "label": c_info["short_topic"]}
        self._render_studio()

    def action_cancel_drag(self) -> None:
        self.drag_state = None
        self._render_studio()

    def action_quick_drop_to_window(self, cname: str, agent_id: str) -> None:
        self.drag_state = None
        self.tui.execute_key_action("dnd", {"op": "lake_to_window", "node_id_or_name": cname, "agent_id": agent_id})
        self._render_studio()

    def action_drop_on_window(self, agent_id: str) -> None:
        if self.drag_state and self.drag_state.get("kind") == "concept":
            cname = self.drag_state["id"]
            self.drag_state = None
            self.tui.execute_key_action("dnd", {"op": "lake_to_window", "node_id_or_name": cname, "agent_id": agent_id})
            self._render_studio()
            return
        if self.drag_state and self.drag_state.get("kind") == "window" and self.drag_state.get("id") != agent_id:
            src_agent = self.drag_state["id"]
            self.drag_state = None
            self.tui.execute_key_action("dnd", {"op": "window_to_window", "source_agent_id": src_agent, "target_agent_id": agent_id})
            self._render_studio()
            return
        # If nothing was picked up yet, pick up the currently selected node from the palace
        fallback_node = self.tui.selected_node_id
        if fallback_node:
            self.tui.execute_key_action("dnd", {"op": "lake_to_window", "node_id_or_name": fallback_node, "agent_id": agent_id})
            self._render_studio()

    def action_drop_on_lake(self) -> None:
        if self.drag_state and self.drag_state.get("kind") == "window":
            aid = self.drag_state["id"]
            self.drag_state = None
            self.tui.execute_key_action("dnd", {"op": "window_to_lake", "agent_id": aid})
            self._render_studio()

    def action_save_window_to_lake(self, agent_id: str) -> None:
        self.drag_state = None
        self.tui.execute_key_action("dnd", {"op": "window_to_lake", "agent_id": agent_id})
        self._render_studio()

    def action_remove_from_window(self, cname: str, agent_id: str) -> None:
        self.tui.execute_key_action("dnd", {"op": "remove_from_window", "node_id_or_name": cname, "agent_id": agent_id})
        self._render_studio()

    def action_sync_windows(self) -> None:
        self.tui.execute_key_action("m", {})
        self._render_studio()

    def action_close(self) -> None:
        self.dismiss(None)


def build_inventory_report(tui: Any) -> Tuple[str, str]:
    """Full untruncated inventory via inventory.LakeInventoryExporter (same as `context-palace list-all`)."""
    import contextlib
    import io

    from context_palace import tui as tui_module

    if tui_module.list_everything is None or tui_module.render_list_all_report is None:
        return ("inventory unavailable", "context_palace.inventory could not be imported")
    with contextlib.redirect_stdout(io.StringIO()):
        inventory = tui_module.list_everything(tui.lake)
        report = tui_module.render_list_all_report(inventory, format="rich")
    summary = inventory.get("summary", {}) if isinstance(inventory, dict) else {}
    title = (
        f"{summary.get('total_nodes', 0):,} nodes · {summary.get('total_edges', 0):,} edges · "
        f"{summary.get('total_sessions', 0):,} sessions · {summary.get('registered_models', 0):,} models"
    )
    return title, report


class ContextPalaceApp(App[None]):
    """Textual Memory Palace control plane with Dark/Light modes and full clickable CRUD."""

    TITLE = "Context Palace"
    REFRESH_SECONDS = 5.0
    ALLOW_SELECT = False

    CSS = f"""
    Screen {{
        background: {DARK_BG};
        color: {DARK_INK};
        padding: 0 1;
    }}
    Screen.theme-light {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
    }}
    #window {{
        background: {DARK_BG};
        border: round {DARK_INK};
        padding: 0 1;
        height: 1fr;
    }}
    Screen.theme-light #window {{
        background: {LIGHT_BG};
        border: round {LIGHT_INK};
    }}
    #header {{
        height: auto;
        margin-bottom: 1;
    }}
    #grid {{
        layout: grid;
        grid-size: 2 2;
        grid-columns: 5fr 3fr;
        grid-rows: 1fr 1.15fr;
        grid-gutter: 0 1;
        height: 1fr;
    }}
    #grid.stacked {{
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: auto;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        scrollbar-color: {DARK_INK_45};
        scrollbar-color-hover: {DARK_INK};
        scrollbar-color-active: {DARK_INK};
        scrollbar-background: {DARK_BG};
        scrollbar-background-hover: {DARK_BG};
        scrollbar-background-active: {DARK_BG};
    }}
    Screen.theme-light #grid.stacked {{
        scrollbar-color: {LIGHT_INK_45};
        scrollbar-color-hover: {LIGHT_INK};
        scrollbar-color-active: {LIGHT_INK};
        scrollbar-background: {LIGHT_BG};
        scrollbar-background-hover: {LIGHT_BG};
        scrollbar-background-active: {LIGHT_BG};
    }}
    #grid.stacked .pane {{
        height: auto;
        max-height: 30;
    }}
    .pane {{
        background: {DARK_BG};
        border: round {DARK_INK};
        padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-gutter: stable;
        scrollbar-color: {DARK_INK_45};
        scrollbar-color-hover: {DARK_INK};
        scrollbar-color-active: {DARK_INK};
        scrollbar-background: {DARK_BG};
        scrollbar-background-hover: {DARK_BG};
        scrollbar-background-active: {DARK_BG};
    }}
    Screen.theme-light .pane {{
        background: {LIGHT_BG};
        border: round {LIGHT_INK};
        scrollbar-color: {LIGHT_INK_45};
        scrollbar-color-hover: {LIGHT_INK};
        scrollbar-color-active: {LIGHT_INK};
        scrollbar-background: {LIGHT_BG};
        scrollbar-background-hover: {LIGHT_BG};
        scrollbar-background-active: {LIGHT_BG};
    }}
    .pane Static {{
        background: {DARK_BG};
        width: 1fr;
    }}
    Screen.theme-light .pane Static {{
        background: {LIGHT_BG};
    }}
    #nl-agent-bar {{
        height: auto;
        border-top: solid {DARK_INK_16};
        padding: 0;
        background: {DARK_BG};
    }}
    Screen.theme-light #nl-agent-bar {{
        border-top: solid {LIGHT_INK_16};
        background: {LIGHT_BG};
    }}
    #nl-agent-input {{
        background: {DARK_BG};
        color: {DARK_INK};
        border: none;
        height: 1;
        padding: 0 1;
    }}
    Screen.theme-light #nl-agent-input {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
    }}
    #statusbar {{
        height: auto;
        border-top: solid {DARK_INK_16};
        padding-top: 0;
    }}
    Screen.theme-light #statusbar {{
        border-top: solid {LIGHT_INK_16};
    }}
    Toast {{
        background: {DARK_BG};
        color: {DARK_INK};
        border-left: wide {DARK_INK};
    }}
    Screen.theme-light Toast {{
        background: {LIGHT_BG};
        color: {LIGHT_INK};
        border-left: wide {LIGHT_INK};
    }}
    """

    BINDINGS = [
        Binding("c", "create", "Create"),
        Binding("r", "read", "Read"),
        Binding("u", "update", "Update"),
        Binding("d", "delete", "Delete"),
        Binding("l", "list_all", "List All"),
        Binding("a", "ask", "Ask"),
        Binding("p", "pin_inject", "Pin & Inject"),
        Binding("m", "models", "Models"),
        Binding("t", "time_travel", "Time-Travel"),
        Binding("o", "cycle_sort_order", "Order"),
        Binding("w", "open_drag_drop_studio", "Drag & Drop"),
        Binding("slash", "focus_nl_agent", "NL Agent", show=False),
        Binding("escape", "blur_nl_agent", "Blur", show=False),
        Binding("k", "toggle_theme", "Theme"),
        Binding("g", "open_crud_hub", "CRUD Hub", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        tui: Any,
        search_query: str = "",
        refresh_seconds: Optional[float] = None,
        theme_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.tui = tui
        if theme_mode in ("dark", "light"):
            self.tui.theme_mode = theme_mode
        elif not hasattr(self.tui, "theme_mode") or self.tui.theme_mode not in ("dark", "light"):
            self.tui.theme_mode = "dark"
        self.search_query = search_query
        self.refresh_seconds = self.REFRESH_SECONDS if refresh_seconds is None else refresh_seconds
        self.last_data: Any = None
        self.pane_renderables: Dict[str, Any] = {}

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        with Container(id="window"):
            yield Static(id="header")
            with Container(id="grid"):
                with VerticalScroll(classes="pane", id="pane-windows"):
                    yield Static(id="windows")
                with VerticalScroll(classes="pane", id="pane-lake"):
                    yield Static(id="lake")
                with VerticalScroll(classes="pane", id="pane-graph"):
                    yield Static(id="graph")
                with VerticalScroll(classes="pane", id="pane-audit"):
                    yield Static(id="audit")
            with Container(id="nl-agent-bar"):
                yield Input(
                    value="",
                    placeholder='palace-agent > Type any update or question in plain English and press Enter (e.g. "Change one_click_install to support brew + curl and set confidence 99%")',
                    id="nl-agent-input",
                )
            yield Static(id="statusbar")

    def on_mount(self) -> None:
        self._sync_theme_class()
        self._apply_breakpoint(self.size.width)
        self.set_focus(None)
        self.refresh_panes()
        if self.refresh_seconds and self.refresh_seconds > 0:
            self.set_interval(self.refresh_seconds, self.refresh_panes)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "nl-agent-input":
            instruction = event.input.value.strip()
            event.input.value = ""
            self.set_focus(None)
            if not instruction:
                return
            self._run(
                "nl",
                {
                    "instruction": instruction,
                    "default_node_name": getattr(self.tui, "selected_node_id", None) or "",
                },
                lambda r: [
                    f"palace-agent > {instruction}",
                    f"  → {r.get('reply', 'Updated')}",
                ],
            )

    def on_resize(self, event: Any) -> None:
        self._apply_breakpoint(event.size.width)
        self.refresh_panes()

    def _find_pane_id_at(self, screen_x: int, screen_y: int) -> Optional[str]:
        try:
            widget, _ = self.get_widget_at(screen_x, screen_y)
        except Exception:
            return None
        cur: Any = widget
        while cur is not None:
            wid = getattr(cur, "id", None)
            if wid in ("pane-windows", "pane-lake", "pane-graph", "pane-audit"):
                return wid
            cur = getattr(cur, "parent", None)
        return None

    def on_mouse_down(self, event: Any) -> None:
        self._mouse_down_pane = self._find_pane_id_at(event.screen_x, event.screen_y)

    def on_mouse_up(self, event: Any) -> None:
        src_pane = getattr(self, "_mouse_down_pane", None)
        self._mouse_down_pane = None
        dst_pane = self._find_pane_id_at(event.screen_x, event.screen_y)
        if not src_pane or not dst_pane or src_pane == dst_pane:
            return
        if src_pane in ("pane-lake", "pane-graph") and dst_pane == "pane-windows":
            target_agent = self._worst_agent()
            if target_agent:
                self.action_drop_on_window(target_agent)
        elif src_pane == "pane-windows" and dst_pane in ("pane-lake", "pane-graph"):
            src_agent = self._worst_agent()
            if src_agent:
                self.action_drag_window_to_lake(src_agent)

    def _sync_theme_class(self) -> None:
        mode = getattr(self.tui, "theme_mode", "dark")
        self.screen.set_class(mode == "light", "theme-light")

    def _apply_breakpoint(self, width: int) -> None:
        grid = self.query_one("#grid")
        grid.set_class(width < STACK_BELOW_COLUMNS, "stacked")

    # -------------------------------------------------------------- live data
    def refresh_panes(self) -> None:
        self._sync_theme_class()
        data = self.tui.collect_dashboard_data(
            search_query=self.search_query,
            frontend="textual",
            size=(self.size.width, self.size.height),
        )
        self.last_data = data
        self.pane_renderables = {
            "header": build_header(data),
            "windows": build_windows_pane(data),
            "lake": build_lake_pane(data),
            "graph": build_graph_pane(data, max_roots=40),
            "audit": build_audit_pane(data),
            "statusbar": build_status_bar(data, width=self.size.width),
        }
        for widget_id, renderable in self.pane_renderables.items():
            self.query_one(f"#{widget_id}", Static).update(renderable)

    # ---------------------------------------------------------------- actions
    def _run(self, key: str, payload: Dict[str, Any], summarize: Callable[[Dict[str, Any]], List[str]]) -> None:
        try:
            result = self.tui.execute_key_action(key, payload)
            lines = summarize(result)
        except Exception as exc:
            self.tui.status_message = f"Error: {exc}"
            lines = [f"palace$ {key} … failed: {exc}"]
        self.tui.last_output = lines
        self.refresh_panes()

    def _prompt(
        self,
        title: str,
        fields: Sequence[Field],
        on_submit: Callable[[Dict[str, str]], None],
        hint: str = "",
    ) -> None:
        def _done(values: Optional[Dict[str, str]]) -> None:
            if values is None:
                self.tui.status_message = "Cancelled"
                self.refresh_panes()
                return
            on_submit(values)

        self.push_screen(
            PromptScreen(title, fields, hint=hint, theme_mode=getattr(self.tui, "theme_mode", "dark")),
            _done,
        )

    def _worst_agent(self) -> str:
        sessions = self.last_data.sessions if self.last_data is not None else []
        if self.tui.selected_agent_id:
            return self.tui.selected_agent_id
        if not sessions:
            return ""
        return max(sessions, key=lambda s: s.utilization_pct).agent_id

    # -------------------------------------------------------- clickable hooks
    def action_hotkey(self, key: str) -> None:
        dispatch = {
            "c": self.action_create,
            "r": self.action_read,
            "u": self.action_update,
            "d": self.action_delete,
            "l": self.action_list_all,
            "a": self.action_ask,
            "p": self.action_pin_inject,
            "m": self.action_models,
            "t": self.action_time_travel,
            "o": self.action_cycle_sort_order,
            "w": self.action_open_drag_drop_studio,
            "k": self.action_toggle_theme,
            "q": self.action_quit,
        }
        fn = dispatch.get(key.lower())
        if fn is not None:
            fn()

    def action_open_drag_drop_studio(self) -> None:
        """Open the split-screen Drag & Drop Context Studio (Centralized SQLite Lake <-> Actual Open Windows)."""
        init_drag = getattr(self, "_main_drag_state", None)
        self._main_drag_state = None
        self.push_screen(
            DragDropStudioScreen(
                self.tui,
                theme_mode=getattr(self.tui, "theme_mode", "dark"),
                initial_drag=init_drag,
            ),
            lambda _: self.refresh_panes(),
        )

    def action_start_drag_concept(self, node_id_or_name: str) -> None:
        """Pick up a concept card from the Centralized SQLite Lake to drop onto an open Agent Window."""
        self.tui.selected_node_id = node_id_or_name
        node = self.tui.lake.get_concept(node_id_or_name, include_tombstoned=True)
        c_info = summarize_concept(node_id_or_name, node.active_value if node else "")
        self._main_drag_state = {
            "kind": "concept",
            "id": node_id_or_name,
            "label": c_info["short_topic"],
        }
        self.tui.status_message = (
            f"✋ Dragging SQLite Concept '{c_info['short_topic']}' ({node_id_or_name}) — Click [⬇ Drop] on any Open Window in WINDOWS!"
        )
        self.refresh_panes()

    def action_drop_on_window(self, agent_id: str) -> None:
        """Drop the currently dragged/selected SQLite concept onto the target Open Agent Window."""
        drag = getattr(self, "_main_drag_state", None)
        cname = (drag.get("id") if drag and drag.get("kind") == "concept" else None) or self.tui.selected_node_id
        self._main_drag_state = None
        if not cname:
            nodes = self.tui.lake.list_concepts(include_tombstoned=False) if hasattr(self.tui.lake, "list_concepts") else []
            cname = nodes[0].canonical_name if nodes else ""
        if not cname:
            self.tui.status_message = "Select or drag a concept from LAKE / GRAPH first"
            self.refresh_panes()
            return
        self._run(
            "dnd",
            {"op": "lake_to_window", "node_id_or_name": cname, "agent_id": agent_id},
            lambda r: [
                f"palace$ dnd {cname} ──► {agent_id}",
                f"  injected '{r.get('short_topic', cname)}' into live window {agent_id}",
            ],
        )

    def action_drag_window_to_lake(self, agent_id: str) -> None:
        """Drag/save an open Agent Window's live topic summary into the Centralized SQLite Lake."""
        self._main_drag_state = None
        self._run(
            "dnd",
            {"op": "window_to_lake", "agent_id": agent_id},
            lambda r: [
                f"palace$ dnd {agent_id} ──► sqlite_lake ({r.get('node', '')})",
                f"  saved '{r.get('short_topic', '')}' into Centralized SQLite Lake",
            ],
        )

    def action_focus_nl_agent(self) -> None:
        try:
            self.query_one("#nl-agent-input", Input).focus()
        except Exception:
            pass

    def action_blur_nl_agent(self) -> None:
        self.set_focus(None)

    def action_select_node(self, node_id_or_name: str) -> None:
        """Select a node shape card and show its plain-English explanation in AUDIT without opening a raw form."""
        self.tui.selected_node_id = node_id_or_name
        self.tui.status_message = (
            f"Selected '{node_id_or_name}' — see plain-English explanation in AUDIT (or click [✎ Form] / type below)"
        )
        self.refresh_panes()

    def action_set_sort_order(self, order: str) -> None:
        """Set the GRAPH viewing order ('pinned_recent', 'recent', 'session', 'category')."""
        self._run(
            "o",
            {"order": order},
            lambda r: [f"palace$ order → {r.get('sort_order', order)}"],
        )

    def action_cycle_sort_order(self) -> None:
        """Cycle the GRAPH viewing order."""
        self._run(
            "o",
            {},
            lambda r: [f"palace$ order → {r.get('sort_order', 'pinned_recent')}"],
        )

    def action_nl_edit_node(self, node_id_or_name: str) -> None:
        """Target a node for natural-language terminal agent editing in the bottom bar."""
        self.tui.selected_node_id = node_id_or_name
        self.tui.status_message = f"Terminal Agent targeting '{node_id_or_name}' — describe what to change and press Enter"
        self.refresh_panes()
        try:
            inp = self.query_one("#nl-agent-input", Input)
            inp.placeholder = (
                f'palace-agent ({node_id_or_name}) > Describe what to change in plain English and press Enter...'
            )
            inp.focus()
        except Exception:
            pass

    def action_toggle_theme(self) -> None:
        self._run(
            "k",
            {},
            lambda r: [f"palace$ k → {r.get('theme_mode', 'dark').upper()} mode"],
        )

    def action_clear_filter(self) -> None:
        self.search_query = ""
        self.tui.selected_agent_id = None
        self.tui.status_message = "Cleared active filters"
        self.refresh_panes()

    def action_filter_concepts(self, query: str) -> None:
        self.search_query = query.strip()
        self.tui.status_message = f"Filter set to '{self.search_query}'" if self.search_query else "Filter cleared"
        self.refresh_panes()

    def action_filter_pinned(self) -> None:
        self.tui.status_message = "Showing pinned concepts at top of GRAPH"
        self.refresh_panes()

    def action_open_concept_crud(self, node_id_or_name: str) -> None:
        """Open the interactive Concept CRUD Inspector Modal for the clicked concept."""
        self.tui.selected_node_id = node_id_or_name

        def _done(res: Optional[Dict[str, Any]]) -> None:
            if not res:
                self.refresh_panes()
                return
            op = res.get("op")
            if op == "create":
                pl = res["payload"]
                self._run(
                    "c",
                    pl,
                    lambda r: [
                        f"palace$ c {pl['canonical_name']}",
                        f"  created {r['node']['node_id']} ({r['node']['category']})",
                    ],
                )
            elif op == "update":
                pl = res["payload"]
                self._run(
                    "u",
                    pl,
                    lambda r: [
                        f"palace$ u {r['node']['canonical_name']}",
                        f"  = {r['node']['active_value']}",
                    ],
                )
            elif op == "delete":
                target = res["node_id_or_name"]
                hard = bool(res.get("hard", False))
                self._run(
                    "d",
                    {"node_id_or_name": target, "hard": hard},
                    lambda r: [
                        f"palace$ d {target}{' --hard' if hard else ''}",
                        f"  deleted={r['deleted']} hard={r['hard']}",
                    ],
                )
            elif op == "pin_inject":
                target = res["node_id_or_name"]
                self._run(
                    "p",
                    {"node_id_or_name": target, "agent_id": self._worst_agent(), "query": target},
                    lambda r: [
                        f"palace$ p {target}",
                        f"  capsule {(r.get('capsule') or {}).get('capsule_tokens', 0):,} tok",
                    ],
                )
            elif op == "time_travel":
                as_of = res.get("as_of")
                self._run("t", {"as_of": as_of}, lambda r: [f"palace$ t {r['as_of']}"])
            elif op == "nl":
                instr = res.get("instruction", "")
                def_node = res.get("default_node_name", "")
                self._run(
                    "nl",
                    {"instruction": instr, "default_node_name": def_node},
                    lambda r: [
                        f"palace-agent ({def_node}) > {instr}",
                        f"  → {r.get('reply', 'Updated')}",
                    ],
                )

        self.push_screen(
            ConceptCrudScreen(
                self.tui,
                node_id_or_name=node_id_or_name,
                is_create=not bool(node_id_or_name),
                theme_mode=getattr(self.tui, "theme_mode", "dark"),
            ),
            _done,
        )

    def action_quick_toggle_pin(self, node_id_or_name: str) -> None:
        existing = self.tui.lake.get_concept(node_id_or_name, include_tombstoned=True)
        new_pin = not bool(getattr(existing, "pinned", False)) if existing else True
        self._run(
            "p",
            {"node_id_or_name": node_id_or_name, "pinned": new_pin},
            lambda r: [f"palace$ pin {node_id_or_name} → pinned={new_pin}"],
        )

    def action_quick_delete_concept(self, node_id_or_name: str) -> None:
        self._run(
            "d",
            {"node_id_or_name": node_id_or_name, "hard": False},
            lambda r: [f"palace$ d {node_id_or_name}", f"  deleted={r['deleted']} hard=False"],
        )

    def action_manage_agent(self, agent_id: str = "") -> None:
        """Open the interactive AgentSession CRUD modal when clicking an agent in WINDOWS or LAKE."""
        if agent_id:
            self.tui.selected_agent_id = agent_id

        def _done(res: Optional[Dict[str, Any]]) -> None:
            if not res:
                self.refresh_panes()
                return
            op = res.get("op")
            if op == "upsert":
                pl = res["payload"]
                self._run(
                    "s",
                    {"op": "upsert", **pl},
                    lambda r: [f"palace$ session upsert {pl['agent_id']}"],
                )
            elif op == "compact":
                pl = res["payload"]
                self.tui.execute_key_action("s", {"op": "upsert", **pl})
                self._run(
                    "p",
                    {"agent_id": pl["agent_id"], "query": "context compaction"},
                    lambda r: [
                        f"palace$ compact {pl['agent_id']}",
                        f"  capsule {(r.get('capsule') or {}).get('capsule_tokens', 0):,} tok",
                    ],
                )
            elif op == "delete":
                aid = res["agent_id"]
                self._run(
                    "s",
                    {"op": "delete", "agent_id": aid},
                    lambda r: [f"palace$ session delete {aid} (deleted={r['deleted']})"],
                )

        self.push_screen(
            AgentCrudScreen(self.tui, agent_id=agent_id, theme_mode=getattr(self.tui, "theme_mode", "dark")),
            _done,
        )

    def action_compact_agent(self, agent_id: str) -> None:
        self.tui.selected_agent_id = agent_id
        self._run(
            "p",
            {"agent_id": agent_id, "query": "compaction"},
            lambda r: [
                f"palace$ p --agent {agent_id}",
                f"  capsule {(r.get('capsule') or {}).get('capsule_tokens', 0):,} tok → {agent_id}",
            ],
        )

    def action_open_crud_hub(self) -> None:
        def _done(res: Optional[Dict[str, Any]]) -> None:
            self.refresh_panes()
            if not res:
                return
            nxt = res.get("next")
            if nxt == "create_concept":
                self.action_open_concept_crud("")
            elif nxt == "edit_concept":
                self.action_open_concept_crud(res.get("name", ""))
            elif nxt == "edit_agent":
                self.action_manage_agent(res.get("agent_id", ""))

        self.push_screen(
            CrudHubScreen(self.tui, theme_mode=getattr(self.tui, "theme_mode", "dark")),
            _done,
        )

    # -------------------------------------------------------- standard hotkeys
    def action_create(self) -> None:
        def submit(v: Dict[str, str]) -> None:
            if not v.get("name"):
                self.tui.status_message = "Create needs a concept name"
                self.refresh_panes()
                return
            self._run(
                "c",
                {
                    "canonical_name": v["name"],
                    "category": v.get("category") or "architecture",
                    "value": v.get("value", ""),
                },
                lambda r: [
                    f"palace$ c {v['name']}",
                    f"  created {r['node']['node_id']} ({r['node']['category']})",
                ],
            )

        self._prompt(
            "[c] Create concept",
            [
                ("name", "Canonical name", ""),
                ("category", "Category", "architecture"),
                ("value", "Active belief value", ""),
            ],
            submit,
            hint="written as an operator_crud revision",
        )

    def action_read(self) -> None:
        def summarize(r: Dict[str, Any]) -> List[str]:
            node = r.get("node")
            if not node:
                return ["palace$ r … not found"]
            lines = [
                f"palace$ r {node['canonical_name']}",
                f"  = {node['active_value']}  (conf {node['confidence']:.2f})",
            ]
            for rev in r.get("history", [])[-3:]:
                state = "active" if rev.get("valid_to") is None else "superseded"
                lines.append(f"  {rev['provenance_type']:<15} {state:<10} {rev['value'][:48]}")
            return lines

        default_target = self.tui.selected_node_id or ""
        self._prompt(
            "[r] Read concept + belief history",
            [("target", "Concept id or name", default_target)],
            lambda v: self._run("r", {"node_id_or_name": v["target"]}, summarize),
        )

    def action_update(self) -> None:
        default_target = self.tui.selected_node_id or ""
        self._prompt(
            "[u] Update belief",
            [("target", "Concept id or name", default_target), ("value", "New belief value", "")],
            lambda v: self._run(
                "u",
                {"node_id_or_name": v["target"], "new_value": v["value"]},
                lambda r: [f"palace$ u {v['target']}", f"  = {r['node']['active_value']}"],
            ),
            hint="supersedes the active BeliefRevision (bitemporal history kept)",
        )

    def action_delete(self) -> None:
        def submit(v: Dict[str, str]) -> None:
            hard = v.get("hard", "").lower() in ("hard", "--hard", "yes", "y")
            self._run(
                "d",
                {"node_id_or_name": v["target"], "hard": hard},
                lambda r: [
                    f"palace$ d {v['target']}{' --hard' if hard else ''}",
                    f"  deleted={r['deleted']} hard={r['hard']}",
                ],
            )

        default_target = self.tui.selected_node_id or ""
        self._prompt(
            "[d] Delete concept",
            [
                ("target", "Concept id or name", default_target),
                ("hard", "Type 'hard' for a hard delete", ""),
            ],
            submit,
            hint="default is a soft tombstone (time-travel still sees it)",
        )

    def action_pin_inject(self) -> None:
        def summarize(r: Dict[str, Any]) -> List[str]:
            cap = r.get("capsule") or {}
            lines = [f"palace$ p {(r.get('node') or {}).get('canonical_name', '')}"]
            if cap:
                lines.append(
                    f"  capsule {cap.get('capsule_tokens', 0):,} tok → {cap.get('target_agent_id', '')}"
                )
            return lines

        def submit(v: Dict[str, str]) -> None:
            payload: Dict[str, Any] = {
                "node_id_or_name": v.get("target") or None,
                "query": v.get("query", ""),
            }
            if v.get("agent"):
                payload["agent_id"] = v["agent"]
            self._run("p", payload, summarize)

        self._prompt(
            "[p] Pin & inject capsule",
            [
                ("target", "Concept to pin (blank = none)", ""),
                ("agent", "Target agent id", self._worst_agent()),
                ("query", "Task query", ""),
            ],
            submit,
            hint="builds context_inject_capsule within the agent's adaptive budget",
        )

    def action_list_all(self) -> None:
        try:
            self.tui.handle_command("l")
            title, report = build_inventory_report(self.tui)
        except Exception as exc:
            self.tui.status_message = f"Error: {exc}"
            self.refresh_panes()
            return
        self.refresh_panes()
        self.push_screen(
            InventoryScreen(title, report, theme_mode=getattr(self.tui, "theme_mode", "dark"))
        )

    def action_ask(self) -> None:
        def submit(v: Dict[str, str]) -> None:
            message = v.get("message", "").strip()
            if not message:
                self.tui.status_message = "Ask needs a message"
                self.refresh_panes()
                return
            try:
                self.tui.handle_command(f"a {message}")
                self.tui.last_output = []
            except Exception as exc:
                self.tui.status_message = f"Error: {exc}"
            self.refresh_panes()
            self.query_one("#pane-audit", VerticalScroll).scroll_end(animate=False)

        self._prompt(
            "[a] Ask the palace",
            [("message", "Message", "")],
            submit,
            hint="questions query the lake · 'remember X is Y' / 'forget X' edit it · answers land in AUDIT",
        )

    def action_models(self) -> None:
        self._run(
            "m",
            {},
            lambda r: [
                "palace$ m",
                f"  {len(r.get('sessions', []))} agent model windows refreshed",
            ],
        )

    def action_time_travel(self) -> None:
        def submit(v: Dict[str, str]) -> None:
            raw = v.get("as_of", "")
            try:
                as_of = float(raw) if raw else None
            except ValueError:
                self.tui.status_message = f"Error: not a unix timestamp: {raw!r}"
                self.refresh_panes()
                return
            self._run(
                "t",
                {"as_of": as_of},
                lambda r: [f"palace$ t {r['as_of'] if r['as_of'] is not None else 'LIVE'}"],
            )

        current = "" if self.tui.as_of is None else str(self.tui.as_of)
        self._prompt(
            "[t] Time-travel",
            [("as_of", "Unix timestamp (blank = LIVE)", current)],
            submit,
            hint="ContextLake.snapshot(as_of) · same as `context-palace time-travel --as-of`",
        )


def run_textual_tui(tui: Any, search_query: str = "", theme_mode: Optional[str] = None) -> None:
    ContextPalaceApp(tui, search_query=search_query, theme_mode=theme_mode).run()
