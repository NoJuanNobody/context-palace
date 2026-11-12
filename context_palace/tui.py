"""
Rich Terminal User Interface (TUI) for Context Palace.
Provides full visibility into Connected Local Agents, Auto-Detected Model Context Windows,
the Bitemporal Knowledge Graph, Relationships & Consensus Audit log, and full CRUD controls.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console, Group  # type: ignore
    from rich.layout import Layout  # type: ignore
    from rich.panel import Panel  # type: ignore
    from rich.progress_bar import ProgressBar  # type: ignore
    from rich.table import Table  # type: ignore
    from rich.text import Text  # type: ignore
except ImportError:
    class Text:  # type: ignore[no-redef]
        def __init__(self, text: str = "", style: str = "") -> None:
            self.text = str(text)
            self.style = style

        def __str__(self) -> str:
            return self.text

    class Group:  # type: ignore[no-redef]
        def __init__(self, *renderables: Any) -> None:
            self.renderables = renderables

        def __str__(self) -> str:
            return "\n".join(str(r) for r in self.renderables)

    class Panel:  # type: ignore[no-redef]
        def __init__(
            self,
            renderable: Any,
            border_style: str = "",
            title: Optional[str] = None,
        ) -> None:
            self.renderable = renderable
            self.border_style = border_style
            self.title = title

        def __str__(self) -> str:
            header = f"╭─ {self.title} ─╮" if self.title else "╭────────────╮"
            body = str(self.renderable)
            return f"{header}\n{body}\n╰────────────╯"

    class Table:  # type: ignore[no-redef]
        def __init__(
            self,
            title: Optional[str] = None,
            expand: bool = True,
            show_lines: bool = False,
        ) -> None:
            self.title = title
            self.expand = expand
            self.show_lines = show_lines
            self.columns: List[str] = []
            self.rows: List[List[str]] = []

        def add_column(self, header: str, **kwargs: Any) -> None:
            self.columns.append(header)

        def add_row(self, *cells: Any) -> None:
            self.rows.append([str(c) for c in cells])

        def __str__(self) -> str:
            lines: List[str] = []
            if self.title:
                lines.append(f"=== {self.title} ===")
            if self.columns:
                lines.append(" | ".join(self.columns))
                lines.append("-" * max(20, len(lines[-1])))
            for row in self.rows:
                lines.append(" | ".join(row))
            return "\n".join(lines)

    class Layout:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class ProgressBar:  # type: ignore[no-redef]
        def __init__(self, total: int = 100, completed: int = 0, width: int = 12) -> None:
            self.total = total
            self.completed = completed
            self.width = width

    class Console:  # type: ignore[no-redef]
        def __init__(
            self,
            record: bool = True,
            width: int = 120,
            force_terminal: bool = False,
        ) -> None:
            self.record = record
            self.width = width
            self.force_terminal = force_terminal
            self._buffer: List[str] = []

        def print(self, *objects: Any) -> None:
            for obj in objects:
                self._buffer.append(str(obj))

        def export_text(self) -> str:
            return "\n\n".join(self._buffer) + "\n"

from context_palace.models import (
    AgentSession,
    BeliefRevision,
    ConceptNode,
    ConflictResolutionEvent,
    ContextCapsule,
    RelationEdge,
    estimate_tokens,
)


def _call_compatible(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a callable while filtering kwargs to accepted parameters if needed."""
    try:
        sig = inspect.signature(fn)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not accepts_var_kw:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return fn(*args, **filtered_kwargs)
    except (ValueError, TypeError):
        pass
    return fn(*args, **kwargs)


class MemoryPalaceTUI:
    """
    Rich TUI Dashboard & Interactive CRUD Inspector for the Federated Context Lake.
    """

    HOTKEY_BAR_TEXT = (
        "[c] Create Concept  [r] Read/History  [u] Update Belief  [d] Delete/Evict  "
        "[p] Pin & Inject Capsule  [m] Refresh Model Windows  [t] Time-Travel  [q] Quit"
    )

    def __init__(
        self,
        lake: Any,
        retriever: Optional[Any] = None,
        registry: Optional[Any] = None,
    ) -> None:
        self.lake = lake
        self.retriever = retriever
        self.registry = registry
        self.as_of: Optional[float] = None
        self.selected_agent_id: Optional[str] = None
        self.status_message: str = "Ready"

    def _fetch_agent_sessions(self) -> List[AgentSession]:
        if hasattr(self.lake, "list_agent_sessions"):
            sessions = self.lake.list_agent_sessions()
            return list(sessions) if sessions else []
        if hasattr(self.lake, "get_agent_sessions"):
            sessions = self.lake.get_agent_sessions()
            return list(sessions) if sessions else []
        return []

    def _fetch_concepts_and_edges(
        self,
        as_of: Optional[float] = None,
        search_query: str = "",
    ) -> tuple[List[ConceptNode], List[RelationEdge], List[ConflictResolutionEvent]]:
        nodes: List[ConceptNode] = []
        edges: List[RelationEdge] = []
        conflicts: List[ConflictResolutionEvent] = []

        if as_of is not None and hasattr(self.lake, "snapshot"):
            snap = _call_compatible(self.lake, as_of=as_of) if callable(self.lake) else _call_compatible(self.lake.snapshot, as_of=as_of)
            if isinstance(snap, dict):
                raw_nodes = snap.get("nodes", snap.get("concepts", []))
                raw_edges = snap.get("edges", snap.get("relations", []))
                raw_conflicts = snap.get("conflicts", snap.get("conflict_events", []))
                for item in raw_nodes:
                    if isinstance(item, ConceptNode):
                        nodes.append(item)
                    elif isinstance(item, dict):
                        nodes.append(
                            ConceptNode(
                                node_id=str(item.get("node_id", "")),
                                canonical_name=str(item.get("canonical_name", "")),
                                category=str(item.get("category", "general")),
                                active_value=str(item.get("active_value", item.get("value", ""))),
                                summary=str(item.get("summary", "")),
                                confidence=float(item.get("confidence", 0.9)),
                                pinned=bool(item.get("pinned", False)),
                                source_agent_id=str(item.get("source_agent_id", "system")),
                                created_at=float(item.get("created_at", 0.0)),
                                updated_at=float(item.get("updated_at", 0.0)),
                                tombstoned=bool(item.get("tombstoned", False)),
                            )
                        )
                for edge in raw_edges:
                    if isinstance(edge, RelationEdge):
                        edges.append(edge)
                    elif isinstance(edge, dict):
                        edges.append(
                            RelationEdge(
                                edge_id=str(edge.get("edge_id", "")),
                                source_id=str(edge.get("source_id", "")),
                                target_id=str(edge.get("target_id", "")),
                                relation_type=str(edge.get("relation_type", "related_to")),
                                weight=float(edge.get("weight", 1.0)),
                                source_agent_id=str(edge.get("source_agent_id", "system")),
                                valid_from=float(edge.get("valid_from", 0.0)),
                                valid_to=edge.get("valid_to"),
                            )
                        )
                for c in raw_conflicts:
                    if isinstance(c, ConflictResolutionEvent):
                        conflicts.append(c)
        else:
            if search_query and hasattr(self.lake, "search_concepts"):
                nodes = list(_call_compatible(self.lake.search_concepts, search_query, limit=50))
            elif hasattr(self.lake, "list_concepts"):
                nodes = list(_call_compatible(self.lake.list_concepts, include_tombstoned=False))

            if hasattr(self.lake, "list_edges"):
                edges = list(_call_compatible(self.lake.list_edges, active_only=True))
            elif hasattr(self.lake, "get_edges"):
                edges = list(_call_compatible(self.lake.get_edges))

        if not conflicts:
            if hasattr(self.lake, "list_conflict_events"):
                conflicts = list(_call_compatible(self.lake.list_conflict_events))
            elif hasattr(self.lake, "get_conflict_events"):
                conflicts = list(_call_compatible(self.lake.get_conflict_events))

        if search_query:
            q_lower = search_query.lower()
            nodes = [
                n
                for n in nodes
                if q_lower in n.canonical_name.lower()
                or q_lower in n.category.lower()
                or q_lower in n.active_value.lower()
                or q_lower in n.summary.lower()
                or q_lower in n.node_id.lower()
            ]

        return nodes, edges, conflicts

    def render_dashboard(
        self,
        as_of: Optional[float] = None,
        selected_agent_id: Optional[str] = None,
        search_query: str = "",
        width: int = 120,
    ) -> str:
        """
        Render the complete Memory Palace TUI into a formatted string.
        """
        effective_as_of = as_of if as_of is not None else self.as_of
        effective_agent = selected_agent_id if selected_agent_id is not None else self.selected_agent_id

        console = Console(record=True, width=width, force_terminal=False)

        # 1. Header Banner
        mode_str = (
            f"TIME-TRAVEL as_of={effective_as_of:.2f}"
            if effective_as_of is not None
            else "LIVE"
        )
        header_text = Text(
            f"CONTEXT PALACE — Federated Memory Lake & TUI CRUD Inspector  [{mode_str}]",
            style="bold cyan",
        )
        sub_line = Text(
            f"Status: {self.status_message}"
            + (f" | Selected Agent: {effective_agent}" if effective_agent else "")
            + (f" | Filter: '{search_query}'" if search_query else ""),
            style="dim",
        )
        header_panel = Panel(
            Group(header_text, sub_line),
            border_style="bright_blue",
            title="Memory Palace Control Plane",
        )

        # 2. Connected Local Agents & Auto-Detected Model Context Windows Table
        agents_table = Table(
            title="Connected Local Agents & Auto-Detected Model Context Windows",
            expand=True,
            show_lines=False,
        )
        agents_table.add_column("Agent ID", style="bold white", no_wrap=True)
        agents_table.add_column("Harness", style="cyan")
        agents_table.add_column("Auto-Detected Model", style="magenta")
        agents_table.add_column("Context Usage (Used / Max)", justify="right")
        agents_table.add_column("Utilization %", justify="right")
        agents_table.add_column("Free Headroom", justify="right")
        agents_table.add_column("Adaptive Capsule Budget", justify="right")
        agents_table.add_column("Status", justify="center")

        sessions = self._fetch_agent_sessions()
        if not sessions:
            agents_table.add_row(
                "(no agents registered)",
                "-",
                "-",
                "0 / 0",
                "0.0%",
                "0",
                "0",
                "IDLE",
            )
        else:
            for sess in sessions:
                status_label = "COMPACT NOW" if sess.needs_compaction else "HEALTHY"
                status_style = "bold red" if sess.needs_compaction else "bold green"
                bar = ProgressBar(total=max(1, sess.max_context_tokens), completed=sess.used_tokens, width=12)
                _ = bar
                agents_table.add_row(
                    sess.agent_id,
                    sess.harness_type,
                    sess.model_id,
                    f"{sess.used_tokens:,} / {sess.max_context_tokens:,}",
                    f"{sess.utilization_pct:.1f}%",
                    f"{sess.remaining_headroom:,}",
                    f"{sess.adaptive_capsule_budget():,}",
                    Text(status_label, style=status_style),
                )

        # 3. Memory Palace Knowledge Graph (CRUD Table)
        nodes, edges, conflicts = self._fetch_concepts_and_edges(
            as_of=effective_as_of, search_query=search_query
        )
        concepts_table = Table(
            title="Memory Palace Knowledge Graph (CRUD Table)",
            expand=True,
            show_lines=False,
        )
        concepts_table.add_column("Node ID", style="dim", no_wrap=True)
        concepts_table.add_column("Canonical Concept", style="bold yellow")
        concepts_table.add_column("Category", style="cyan")
        concepts_table.add_column("Active Belief Value", style="white")
        concepts_table.add_column("Confidence", justify="right")
        concepts_table.add_column("Source Agent", style="magenta")
        concepts_table.add_column("Pinned", justify="center")

        if not nodes:
            concepts_table.add_row(
                "-",
                "(no active concepts)",
                "-",
                "-",
                "-",
                "-",
                "-",
            )
        else:
            for node in nodes:
                pin_marker = "PINNED" if node.pinned else "-"
                concepts_table.add_row(
                    node.node_id,
                    node.canonical_name,
                    node.category,
                    node.active_value,
                    f"{node.confidence:.2f}",
                    node.source_agent_id,
                    pin_marker,
                )

        # 4. Relationships & Consensus Audit Panel
        node_name_by_id = {n.node_id: n.canonical_name for n in nodes}
        rel_lines: List[str] = []
        if edges:
            for edge in edges[:12]:
                src_label = node_name_by_id.get(edge.source_id, edge.source_id)
                dst_label = node_name_by_id.get(edge.target_id, edge.target_id)
                rel_lines.append(
                    f"Edge: {src_label} -> [{edge.relation_type}] -> {dst_label} (w={edge.weight:.2f}, via {edge.source_agent_id})"
                )
        else:
            rel_lines.append("Edges: (no active relation edges)")

        if conflicts:
            for ev in conflicts[:8]:
                rel_lines.append(
                    f"Conflict Audit [{ev.event_id}]: node={ev.node_id} winner={ev.winning_agent_id} "
                    f"({ev.winning_score:.2f}) over {ev.losing_agent_id} ({ev.losing_score:.2f}) — {ev.reason}"
                )
        else:
            rel_lines.append("Consensus Audit: 0 unresolved conflicts")

        audit_panel = Panel(
            "\n".join(rel_lines),
            title="Relationships & Consensus Audit Panel",
            border_style="magenta",
        )

        # 5. Interactive CRUD Hotkey Bar
        hotkey_panel = Panel(
            Text(self.HOTKEY_BAR_TEXT, style="bold white"),
            title="Interactive CRUD Hotkey Bar",
            border_style="green",
        )

        _layout = Layout()
        console.print(header_panel)
        console.print(agents_table)
        console.print(concepts_table)
        console.print(audit_panel)
        console.print(hotkey_panel)

        return console.export_text()

    def execute_key_action(
        self,
        key: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute deterministic CRUD or control actions in the Memory Palace.
        Supported keys:
          - 'c': Create Concept
          - 'r': Read Concept & Belief History
          - 'u': Update Concept Belief
          - 'd': Delete / Tombstone / Hard-Evict Concept
          - 'p': Pin Concept & Build/Inject Context Capsule
          - 'm': Refresh Auto-Detected Model Windows for Local Agents
          - 't': Time-Travel Snapshot
        """
        payload = payload or {}
        action_key = key.strip().lower()

        if action_key == "c":
            canonical_name = payload["canonical_name"]
            category = payload.get("category", "architecture")
            value = payload.get("value", payload.get("active_value", ""))
            summary = payload.get("summary", "")
            confidence = float(payload.get("confidence", 0.9))
            source_agent_id = payload.get("source_agent_id", "operator_tui")
            provenance_type = payload.get("provenance_type", "operator_crud")
            pinned = bool(payload.get("pinned", False))
            timestamp = payload.get("timestamp")

            kwargs: Dict[str, Any] = {
                "canonical_name": canonical_name,
                "category": category,
                "value": value,
                "summary": summary,
                "confidence": confidence,
                "source_agent_id": source_agent_id,
                "provenance_type": provenance_type,
                "pinned": pinned,
            }
            if timestamp is not None:
                kwargs["timestamp"] = float(timestamp)

            node = _call_compatible(self.lake.create_concept, **kwargs)
            self.status_message = f"Created concept '{canonical_name}'"
            return {
                "status": "ok",
                "action": "create",
                "node": node.to_dict() if hasattr(node, "to_dict") else node,
            }

        if action_key == "r":
            identifier = payload.get("node_id_or_name") or payload.get("canonical_name") or payload.get("node_id")
            if not identifier:
                raise ValueError("Missing 'node_id_or_name' in payload for read action 'r'.")
            node = self.lake.get_concept(identifier)
            history: List[BeliefRevision] = []
            if hasattr(self.lake, "get_belief_history"):
                history = list(self.lake.get_belief_history(identifier))
            self.status_message = f"Read concept '{identifier}' ({len(history)} revisions)"
            return {
                "status": "ok",
                "action": "read",
                "node": node.to_dict() if (node and hasattr(node, "to_dict")) else node,
                "history": [
                    rev.to_dict() if hasattr(rev, "to_dict") else rev for rev in history
                ],
            }

        if action_key == "u":
            identifier = payload.get("node_id_or_name") or payload.get("canonical_name") or payload.get("node_id")
            if not identifier:
                raise ValueError("Missing 'node_id_or_name' in payload for update action 'u'.")
            new_value = payload.get("new_value", payload.get("value", ""))
            confidence = float(payload.get("confidence", 0.95))
            source_agent_id = payload.get("source_agent_id", "operator_tui")
            provenance_type = payload.get("provenance_type", "operator_crud")
            summary = payload.get("summary")
            pinned = payload.get("pinned")
            timestamp = payload.get("timestamp")

            update_kwargs: Dict[str, Any] = {
                "node_id_or_name": identifier,
                "new_value": new_value,
                "confidence": confidence,
                "source_agent_id": source_agent_id,
                "provenance_type": provenance_type,
            }
            if summary is not None:
                update_kwargs["summary"] = summary
            if pinned is not None:
                update_kwargs["pinned"] = bool(pinned)
            if timestamp is not None:
                update_kwargs["timestamp"] = float(timestamp)

            updated = _call_compatible(self.lake.update_concept, **update_kwargs)
            self.status_message = f"Updated concept '{identifier}'"
            return {
                "status": "ok",
                "action": "update",
                "node": updated.to_dict() if hasattr(updated, "to_dict") else updated,
            }

        if action_key == "d":
            identifier = payload.get("node_id_or_name") or payload.get("canonical_name") or payload.get("node_id")
            if not identifier:
                raise ValueError("Missing 'node_id_or_name' in payload for delete action 'd'.")
            hard = bool(payload.get("hard", False))
            deleted = _call_compatible(self.lake.delete_concept, identifier, hard=hard)
            self.status_message = f"Deleted concept '{identifier}' (hard={hard})"
            return {
                "status": "ok",
                "action": "delete",
                "deleted": bool(deleted),
                "hard": hard,
            }

        if action_key == "p":
            identifier = payload.get("node_id_or_name") or payload.get("canonical_name") or payload.get("node_id")
            pinned_flag = bool(payload.get("pinned", True))
            pinned_node = None
            if identifier:
                if hasattr(self.lake, "pin_concept"):
                    pinned_node = _call_compatible(self.lake.pin_concept, identifier, pinned=pinned_flag)
                else:
                    existing = self.lake.get_concept(identifier)
                    if existing is not None:
                        pinned_node = _call_compatible(
                            self.lake.update_concept,
                            node_id_or_name=identifier,
                            new_value=existing.active_value,
                            pinned=pinned_flag,
                        )

            capsule_dict = None
            sidecar_path_res = None
            target_agent_id = payload.get("agent_id") or self.selected_agent_id
            query = payload.get("query", "")
            sidecar_path = payload.get("sidecar_path")

            if self.retriever is not None and target_agent_id:
                session_obj = None
                if hasattr(self.lake, "get_agent_session"):
                    session_obj = self.lake.get_agent_session(target_agent_id)
                if session_obj is None:
                    session_obj = AgentSession(
                        agent_id=target_agent_id,
                        harness_type="antigravity",
                        model_id="gemini-4-argon",
                        max_context_tokens=1_000_000,
                        used_tokens=50_000,
                    )
                capsule = _call_compatible(
                    self.retriever.build_context_capsule,
                    session=session_obj,
                    task_query=query,
                    agent_id=target_agent_id,
                    query=query,
                    as_of=self.as_of,
                )
                if sidecar_path and hasattr(self.retriever, "inject_into_sidecar"):
                    self.retriever.inject_into_sidecar(capsule, sidecar_path)
                    sidecar_path_res = str(sidecar_path)
                capsule_dict = capsule.to_dict() if hasattr(capsule, "to_dict") else capsule

            self.status_message = f"Pinned '{identifier or ''}' & prepared capsule"
            return {
                "status": "ok",
                "action": "pin_and_inject",
                "node": pinned_node.to_dict() if (pinned_node and hasattr(pinned_node, "to_dict")) else pinned_node,
                "capsule": capsule_dict,
                "sidecar_path": sidecar_path_res,
            }

        if action_key == "m":
            discovered: List[AgentSession] = []
            if self.registry is not None and hasattr(self.registry, "discover_local_agents"):
                search_dirs = payload.get("search_dirs")
                limit = int(payload.get("limit", 10))
                discovered = list(
                    _call_compatible(
                        self.registry.discover_local_agents,
                        search_dirs=search_dirs,
                        limit=limit,
                    )
                )
                if hasattr(self.lake, "upsert_agent_session"):
                    for sess in discovered:
                        self.lake.upsert_agent_session(sess)
            else:
                discovered = self._fetch_agent_sessions()

            self.status_message = f"Refreshed {len(discovered)} agent model windows"
            return {
                "status": "ok",
                "action": "refresh_models",
                "sessions": [s.to_dict() if hasattr(s, "to_dict") else s for s in discovered],
            }

        if action_key == "t":
            self.as_of = float(payload["as_of"]) if payload.get("as_of") is not None else None
            snap: Dict[str, Any] = {}
            if hasattr(self.lake, "snapshot"):
                raw_snap = _call_compatible(self.lake.snapshot, as_of=self.as_of)
                if isinstance(raw_snap, dict):
                    snap = {
                        k: [item.to_dict() if hasattr(item, "to_dict") else item for item in v]
                        if isinstance(v, list)
                        else v
                        for k, v in raw_snap.items()
                    }
            mode = f"as_of={self.as_of}" if self.as_of is not None else "LIVE"
            self.status_message = f"Time-travel set to {mode}"
            return {
                "status": "ok",
                "action": "time_travel",
                "as_of": self.as_of,
                "snapshot": snap,
            }

        raise ValueError(f"Unsupported key action: {key!r}")

    def run_interactive(self) -> None:
        """
        Render the dashboard and run a terminal command loop.
        """
        while True:
            print(self.render_dashboard())
            try:
                raw = input("context-palace [c/r/u/d/p/m/t/q]> ").strip()
            except EOFError:
                break
            if not raw or raw.lower() == "q":
                break
            parts = raw.split()
            cmd = parts[0].lower()
            try:
                if cmd == "c":
                    name = parts[1] if len(parts) > 1 else input("Canonical concept name: ").strip()
                    cat = parts[2] if len(parts) > 2 else input("Category [architecture]: ").strip() or "architecture"
                    val = " ".join(parts[3:]) if len(parts) > 3 else input("Active belief value: ").strip()
                    self.execute_key_action("c", {"canonical_name": name, "category": cat, "value": val})
                elif cmd == "r":
                    target = parts[1] if len(parts) > 1 else input("Concept ID or name: ").strip()
                    res = self.execute_key_action("r", {"node_id_or_name": target})
                    print(res)
                elif cmd == "u":
                    target = parts[1] if len(parts) > 1 else input("Concept ID or name: ").strip()
                    val = " ".join(parts[2:]) if len(parts) > 2 else input("New belief value: ").strip()
                    self.execute_key_action("u", {"node_id_or_name": target, "new_value": val})
                elif cmd == "d":
                    target = parts[1] if len(parts) > 1 else input("Concept ID or name: ").strip()
                    self.execute_key_action("d", {"node_id_or_name": target})
                elif cmd == "p":
                    target = parts[1] if len(parts) > 1 else input("Concept ID or name to pin: ").strip()
                    self.execute_key_action("p", {"node_id_or_name": target})
                elif cmd == "m":
                    self.execute_key_action("m", {})
                elif cmd == "t":
                    ts_str = parts[1] if len(parts) > 1 else input("Timestamp (empty for LIVE): ").strip()
                    ts_val = float(ts_str) if ts_str else None
                    self.execute_key_action("t", {"as_of": ts_val})
            except Exception as exc:
                self.status_message = f"Error: {exc}"
