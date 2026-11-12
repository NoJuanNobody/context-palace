"""
Interactive Textual front-end for the Context Palace v2 TUI.

Layout (from the v2 UI brief):

    header          time · window chrome · title · LIVE / TIME-TRAVEL
    row_top         WINDOWS (1.25)  | LAKE  (0.75)
    row_bottom      GRAPH   (1.25)  | AUDIT (0.75)
    status bar      mode · db path · [c][r][u][d][p][m][t][q] · textual · WxH · UTF-8

Every pane is rendered by ``context_palace.tui_view`` from live data collected by
``MemoryPalaceTUI.collect_dashboard_data``; every hotkey calls
``MemoryPalaceTUI.execute_key_action`` (the same code path as the CLI ``crud`` commands).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Static

from context_palace.tui_view import (
    BG,
    INK,
    INK_16,
    INK_45,
    STACK_BELOW_COLUMNS,
    build_audit_pane,
    build_graph_pane,
    build_header,
    build_lake_pane,
    build_status_bar,
    build_windows_pane,
)

Field = Tuple[str, str, str]  # (key, label, default)


class PromptScreen(ModalScreen[Optional[Dict[str, str]]]):
    """Small hairline modal that collects the arguments for one hotkey action."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = f"""
    PromptScreen {{
        align: center middle;
        background: {BG} 60%;
    }}
    PromptScreen > Vertical {{
        width: 72;
        height: auto;
        background: {BG};
        color: {INK};
        border: round {INK};
        padding: 1 2;
    }}
    PromptScreen .prompt-title {{
        color: {INK};
        text-style: bold;
        margin-bottom: 1;
    }}
    PromptScreen .prompt-label {{
        color: {INK_45};
    }}
    PromptScreen .prompt-hint {{
        color: {INK_45};
        margin-top: 1;
    }}
    PromptScreen Input {{
        background: {BG};
        color: {INK};
        border: round {INK_45};
        padding: 0 1;
    }}
    PromptScreen Input:focus {{
        border: round {INK};
    }}
    """

    def __init__(self, title: str, fields: Sequence[Field], hint: str = "") -> None:
        super().__init__()
        self.title_text = title
        self.fields = list(fields)
        self.hint = hint

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.title_text, classes="prompt-title")
            for key, label, default in self.fields:
                yield Label(label.upper(), classes="prompt-label")
                yield Input(value=default, id=f"field-{key}")
            yield Label(
                (self.hint + "\n" if self.hint else "") + "enter = next / submit · esc = cancel",
                classes="prompt-hint",
            )

    def on_mount(self) -> None:
        self.query(Input).first().focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        inputs = list(self.query(Input))
        idx = inputs.index(event.input)
        if idx < len(inputs) - 1:
            inputs[idx + 1].focus()
            return
        self.dismiss({key: self.query_one(f"#field-{key}", Input).value.strip() for key, _, _ in self.fields})

    def action_cancel(self) -> None:
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
        background: {BG};
        padding: 1 2;
    }}
    InventoryScreen > VerticalScroll {{
        background: {BG};
        color: {INK};
        border: round {INK};
        padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-color: {INK_45};
        scrollbar-color-hover: {INK};
        scrollbar-color-active: {INK};
        scrollbar-background: {BG};
        scrollbar-background-hover: {BG};
        scrollbar-background-active: {BG};
    }}
    InventoryScreen Static {{
        background: {BG};
        color: {INK};
    }}
    InventoryScreen .inv-title {{
        text-style: bold;
    }}
    InventoryScreen .inv-hint {{
        color: {INK_45};
    }}
    """

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.title_text = title
        self.body = body

    def compose(self) -> ComposeResult:
        from rich.text import Text

        with VerticalScroll(id="inventory"):
            yield Static("LIST ALL  ·  " + self.title_text, classes="inv-title")
            yield Static("esc / q / l = back to the palace", classes="inv-hint")
            yield Static(Text(self.body), id="inventory-body")

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
    """Textual Memory Palace control plane."""

    TITLE = "Context Palace"
    REFRESH_SECONDS = 5.0

    CSS = f"""
    Screen {{
        background: {BG};
        color: {INK};
        padding: 0 1;
    }}
    #window {{
        background: {BG};
        border: round {INK};
        padding: 0 1;
        height: 1fr;
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
        scrollbar-color: {INK_45};
        scrollbar-color-hover: {INK};
        scrollbar-color-active: {INK};
        scrollbar-background: {BG};
        scrollbar-background-hover: {BG};
        scrollbar-background-active: {BG};
    }}
    #grid.stacked .pane {{
        height: auto;
        max-height: 30;
    }}
    .pane {{
        background: {BG};
        border: round {INK};
        padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-gutter: stable;
        scrollbar-color: {INK_45};
        scrollbar-color-hover: {INK};
        scrollbar-color-active: {INK};
        scrollbar-background: {BG};
        scrollbar-background-hover: {BG};
        scrollbar-background-active: {BG};
    }}
    .pane Static {{
        background: {BG};
        width: 1fr;
    }}
    #statusbar {{
        height: auto;
        border-top: solid {INK_16};
        padding-top: 0;
    }}
    Toast {{
        background: {BG};
        color: {INK};
        border-left: wide {INK};
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
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, tui: Any, search_query: str = "", refresh_seconds: Optional[float] = None) -> None:
        super().__init__()
        self.tui = tui
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
            yield Static(id="statusbar")

    def on_mount(self) -> None:
        self._apply_breakpoint(self.size.width)
        self.refresh_panes()
        if self.refresh_seconds and self.refresh_seconds > 0:
            self.set_interval(self.refresh_seconds, self.refresh_panes)

    def on_resize(self, event: Any) -> None:
        self._apply_breakpoint(event.size.width)
        self.refresh_panes()

    def _apply_breakpoint(self, width: int) -> None:
        grid = self.query_one("#grid")
        grid.set_class(width < STACK_BELOW_COLUMNS, "stacked")

    # -------------------------------------------------------------- live data
    def refresh_panes(self) -> None:
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
        except Exception as exc:  # surface lake errors in the AUDIT prompt line
            self.tui.status_message = f"Error: {exc}"
            lines = [f"palace$ {key} … failed: {exc}"]
        self.tui.last_output = lines
        self.refresh_panes()

    def _prompt(self, title: str, fields: Sequence[Field], on_submit: Callable[[Dict[str, str]], None], hint: str = "") -> None:
        def _done(values: Optional[Dict[str, str]]) -> None:
            if values is None:
                self.tui.status_message = "Cancelled"
                self.refresh_panes()
                return
            on_submit(values)

        self.push_screen(PromptScreen(title, fields, hint=hint), _done)

    def _worst_agent(self) -> str:
        sessions = self.last_data.sessions if self.last_data is not None else []
        if self.tui.selected_agent_id:
            return self.tui.selected_agent_id
        if not sessions:
            return ""
        return max(sessions, key=lambda s: s.utilization_pct).agent_id

    def action_create(self) -> None:
        def submit(v: Dict[str, str]) -> None:
            if not v.get("name"):
                self.tui.status_message = "Create needs a concept name"
                self.refresh_panes()
                return
            self._run(
                "c",
                {"canonical_name": v["name"], "category": v.get("category") or "architecture", "value": v.get("value", "")},
                lambda r: [f"palace$ c {v['name']}", f"  created {r['node']['node_id']} ({r['node']['category']})"],
            )

        self._prompt(
            "[c] Create concept",
            [("name", "Canonical name", ""), ("category", "Category", "architecture"), ("value", "Active belief value", "")],
            submit,
            hint="written as an operator_crud revision",
        )

    def action_read(self) -> None:
        def summarize(r: Dict[str, Any]) -> List[str]:
            node = r.get("node")
            if not node:
                return ["palace$ r … not found"]
            lines = [f"palace$ r {node['canonical_name']}", f"  = {node['active_value']}  (conf {node['confidence']:.2f})"]
            for rev in r.get("history", [])[-3:]:
                state = "active" if rev.get("valid_to") is None else "superseded"
                lines.append(f"  {rev['provenance_type']:<15} {state:<10} {rev['value'][:48]}")
            return lines

        self._prompt(
            "[r] Read concept + belief history",
            [("target", "Concept id or name", "")],
            lambda v: self._run("r", {"node_id_or_name": v["target"]}, summarize),
        )

    def action_update(self) -> None:
        self._prompt(
            "[u] Update belief",
            [("target", "Concept id or name", ""), ("value", "New belief value", "")],
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
                lambda r: [f"palace$ d {v['target']}{' --hard' if hard else ''}", f"  deleted={r['deleted']} hard={r['hard']}"],
            )

        self._prompt(
            "[d] Delete concept",
            [("target", "Concept id or name", ""), ("hard", "Type 'hard' for a hard delete", "")],
            submit,
            hint="default is a soft tombstone (time-travel still sees it)",
        )

    def action_pin_inject(self) -> None:
        def summarize(r: Dict[str, Any]) -> List[str]:
            cap = r.get("capsule") or {}
            lines = [f"palace$ p {(r.get('node') or {}).get('canonical_name', '')}"]
            if cap:
                lines.append(f"  capsule {cap.get('capsule_tokens', 0):,} tok → {cap.get('target_agent_id', '')}")
            return lines

        def submit(v: Dict[str, str]) -> None:
            payload: Dict[str, Any] = {"node_id_or_name": v.get("target") or None, "query": v.get("query", "")}
            if v.get("agent"):
                payload["agent_id"] = v["agent"]
            self._run("p", payload, summarize)

        self._prompt(
            "[p] Pin & inject capsule",
            [("target", "Concept to pin (blank = none)", ""), ("agent", "Target agent id", self._worst_agent()), ("query", "Task query", "")],
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
        self.push_screen(InventoryScreen(title, report))

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
        self._run("m", {}, lambda r: ["palace$ m", f"  {len(r.get('sessions', []))} agent model windows refreshed"])

    def action_time_travel(self) -> None:
        def submit(v: Dict[str, str]) -> None:
            raw = v.get("as_of", "")
            try:
                as_of = float(raw) if raw else None
            except ValueError:
                self.tui.status_message = f"Error: not a unix timestamp: {raw!r}"
                self.refresh_panes()
                return
            self._run("t", {"as_of": as_of}, lambda r: [f"palace$ t {r['as_of'] if r['as_of'] is not None else 'LIVE'}"])

        current = "" if self.tui.as_of is None else str(self.tui.as_of)
        self._prompt(
            "[t] Time-travel",
            [("as_of", "Unix timestamp (blank = LIVE)", current)],
            submit,
            hint="ContextLake.snapshot(as_of) · same as `context-palace time-travel --as-of`",
        )


def run_textual_tui(tui: Any, search_query: str = "") -> None:
    ContextPalaceApp(tui, search_query=search_query).run()
