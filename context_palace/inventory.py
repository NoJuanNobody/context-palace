"""
Comprehensive Lake Inventory and Multi-Format List-All Exporter for Context Palace.
Provides zero-truncation export across all 5 lake tables:
  1. memory_nodes (active + superseded, with full untruncated content)
  2. memory_edges
  3. agent_sessions
  4. extraction_watermarks
  5. model_registry
Supports JSON, clean Markdown, and Rich terminal rendering with zero ellipsis or truncation.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional

from context_palace.lake import ContextLake
from context_palace.models import estimate_tokens
from context_palace.model_registry import ModelContextRegistry

# Ensure ModelContextRegistry has list_models alias if not present
if hasattr(ModelContextRegistry, "list_known_models") and not hasattr(ModelContextRegistry, "list_models"):
    ModelContextRegistry.list_models = ModelContextRegistry.list_known_models

# Graceful Rich fallback matching tui.py for zero-dependency execution
try:
    from rich.console import Console, Group  # type: ignore
    from rich.panel import Panel  # type: ignore
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
            box: Any = None,
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


class LakeInventoryExporter:
    """
    Exports a comprehensive inventory of all tables in the Federated Context Lake
    and model context registry with zero content truncation.
    """

    def __init__(
        self,
        lake: ContextLake,
        registry: Optional[ModelContextRegistry] = None,
    ) -> None:
        self.lake = lake
        self.registry = registry
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure schema compatibility for memory_nodes, memory_edges, extraction_watermarks, model_registry."""
        cur = self.lake._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_nodes (
                node_id TEXT,
                memory_type TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'standard',
                source_agent TEXT NOT NULL DEFAULT 'system',
                confidence REAL NOT NULL DEFAULT 0.9,
                valid_from REAL NOT NULL,
                valid_until REAL,
                superseded_by TEXT,
                content TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_edges (
                edge_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                source_agent TEXT NOT NULL DEFAULT 'system',
                valid_from REAL NOT NULL,
                valid_until REAL
            );

            CREATE TABLE IF NOT EXISTS extraction_watermarks (
                agent_id TEXT PRIMARY KEY,
                transcript_path TEXT,
                last_turn INTEGER NOT NULL DEFAULT 0,
                last_offset INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS model_registry (
                model_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                max_context_tokens INTEGER NOT NULL,
                max_output_tokens INTEGER NOT NULL DEFAULT 8192,
                source TEXT NOT NULL DEFAULT 'db'
            );
            """
        )
        self.lake._conn.commit()

    def _fetch_all_nodes(self) -> List[Dict[str, Any]]:
        """Fetch all nodes from memory_nodes and concept_nodes/belief_revisions without truncation."""
        all_nodes: List[Dict[str, Any]] = []
        seen_keys = set()

        # 1. From memory_nodes table
        try:
            cur = self.lake._conn.execute(
                "SELECT * FROM memory_nodes ORDER BY valid_from DESC, node_id ASC"
            )
            for r in cur.fetchall():
                d = dict(r)
                node_id = str(d.get("node_id", ""))
                v_from = float(d.get("valid_from", 0.0))
                seen_keys.add((node_id, v_from))
                all_nodes.append({
                    "node_id": node_id,
                    "memory_type": str(d.get("memory_type", "general")),
                    "tier": str(d.get("tier", "standard")),
                    "source_agent": str(d.get("source_agent", "system")),
                    "confidence": float(d.get("confidence", 0.9)),
                    "valid_from": v_from,
                    "valid_until": float(d["valid_until"]) if d.get("valid_until") is not None else None,
                    "superseded_by": d.get("superseded_by"),
                    "content": str(d.get("content", "")),
                })
        except Exception:
            pass

        # 2. Also incorporate from belief_revisions + concept_nodes if present
        try:
            cur = self.lake._conn.execute(
                """
                SELECT br.revision_id, br.node_id, br.value, br.confidence, br.source_agent_id,
                       br.provenance_type, br.turn_id, br.valid_from, br.valid_to, br.superseded_by,
                       cn.category, cn.pinned, cn.canonical_name, cn.summary
                FROM belief_revisions br
                LEFT JOIN concept_nodes cn ON br.node_id = cn.node_id
                ORDER BY br.valid_from DESC, br.node_id ASC
                """
            )
            for r in cur.fetchall():
                d = dict(r)
                node_id = str(d.get("node_id", ""))
                v_from = float(d.get("valid_from", 0.0))
                if (node_id, v_from) in seen_keys:
                    continue
                seen_keys.add((node_id, v_from))
                category = d.get("category") or d.get("provenance_type") or "concept"
                pinned = bool(d.get("pinned", 0))
                all_nodes.append({
                    "node_id": node_id,
                    "memory_type": category,
                    "tier": "pinned" if pinned else "standard",
                    "source_agent": str(d.get("source_agent_id", "system")),
                    "confidence": float(d.get("confidence", 0.9)),
                    "valid_from": v_from,
                    "valid_until": float(d["valid_to"]) if d.get("valid_to") is not None else None,
                    "superseded_by": d.get("superseded_by"),
                    "content": str(d.get("value", "")),
                })
        except Exception:
            pass

        return all_nodes

    def _fetch_all_edges(self) -> List[Dict[str, Any]]:
        """Fetch all edges from memory_edges and relation_edges."""
        all_edges: List[Dict[str, Any]] = []
        seen_edges = set()

        try:
            cur = self.lake._conn.execute(
                "SELECT * FROM memory_edges ORDER BY valid_from ASC, edge_id ASC"
            )
            for r in cur.fetchall():
                d = dict(r)
                eid = str(d.get("edge_id", ""))
                seen_edges.add(eid)
                all_edges.append({
                    "edge_id": eid,
                    "source_id": str(d.get("source_id", "")),
                    "target_id": str(d.get("target_id", "")),
                    "relation_type": str(d.get("relation_type", "")),
                    "weight": float(d.get("weight", 1.0)),
                    "source_agent": str(d.get("source_agent", "system")),
                    "valid_from": float(d.get("valid_from", 0.0)),
                    "valid_until": float(d["valid_until"]) if d.get("valid_until") is not None else None,
                })
        except Exception:
            pass

        try:
            cur = self.lake._conn.execute(
                "SELECT * FROM relation_edges ORDER BY valid_from ASC, edge_id ASC"
            )
            for r in cur.fetchall():
                d = dict(r)
                eid = str(d.get("edge_id", ""))
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                all_edges.append({
                    "edge_id": eid,
                    "source_id": str(d.get("source_id", "")),
                    "target_id": str(d.get("target_id", "")),
                    "relation_type": str(d.get("relation_type", "")),
                    "weight": float(d.get("weight", 1.0)),
                    "source_agent": str(d.get("source_agent_id", "system")),
                    "valid_from": float(d.get("valid_from", 0.0)),
                    "valid_until": float(d["valid_to"]) if d.get("valid_to") is not None else None,
                })
        except Exception:
            pass

        all_edges.sort(key=lambda e: (float(e.get("valid_from") or 0.0), str(e.get("edge_id") or "")))
        return all_edges

    def _fetch_agent_sessions(self) -> List[Dict[str, Any]]:
        """Fetch all connected agent sessions."""
        sessions: List[Dict[str, Any]] = []
        try:
            cur = self.lake._conn.execute(
                "SELECT * FROM agent_sessions ORDER BY updated_at DESC, agent_id ASC"
            )
            for r in cur.fetchall():
                d = dict(r)
                sessions.append({
                    "agent_id": str(d.get("agent_id", "")),
                    "harness_type": str(d.get("harness_type", "")),
                    "model_id": str(d.get("model_id", "")),
                    "max_context_tokens": int(d.get("max_context_tokens", 0)),
                    "used_tokens": int(d.get("used_tokens", 0)),
                    "transcript_path": d.get("transcript_path"),
                    "last_processed_turn": int(d.get("last_processed_turn", 0)),
                    "last_byte_offset": int(d.get("last_byte_offset", 0)),
                    "updated_at": float(d.get("updated_at", 0.0)),
                    "metadata_json": str(d.get("metadata_json", "{}")),
                })
        except Exception:
            pass
        return sessions

    def _fetch_watermarks(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch extraction watermarks, falling back to agent_sessions if table is empty."""
        watermarks: List[Dict[str, Any]] = []
        try:
            cur = self.lake._conn.execute(
                "SELECT * FROM extraction_watermarks ORDER BY updated_at DESC, agent_id ASC"
            )
            for r in cur.fetchall():
                d = dict(r)
                watermarks.append({
                    "agent_id": str(d.get("agent_id", "")),
                    "transcript_path": d.get("transcript_path"),
                    "last_turn": int(d.get("last_turn", d.get("last_processed_turn", 0))),
                    "last_offset": int(d.get("last_offset", d.get("last_byte_offset", 0))),
                    "updated_at": float(d.get("updated_at", 0.0)),
                })
        except Exception:
            pass

        if not watermarks:
            for s in sessions:
                if s.get("last_processed_turn", 0) > 0 or s.get("last_byte_offset", 0) > 0 or s.get("transcript_path"):
                    watermarks.append({
                        "agent_id": s["agent_id"],
                        "transcript_path": s.get("transcript_path"),
                        "last_turn": s.get("last_processed_turn", 0),
                        "last_offset": s.get("last_byte_offset", 0),
                        "updated_at": s.get("updated_at", 0.0),
                    })
        return watermarks

    def _fetch_models(self) -> List[Dict[str, Any]]:
        """Combine ModelContextRegistry models and database model_registry rows."""
        models_by_id: Dict[str, Dict[str, Any]] = {}

        # 1. From registry if provided
        if self.registry is not None:
            list_fn = getattr(self.registry, "list_models", getattr(self.registry, "list_known_models", None))
            if callable(list_fn):
                for spec in list_fn():
                    if hasattr(spec, "to_dict"):
                        m_dict = spec.to_dict()
                    elif isinstance(spec, dict):
                        m_dict = dict(spec)
                    else:
                        m_dict = dataclasses.asdict(spec)
                    mid = str(m_dict.get("model_id", "")).strip().lower()
                    if mid:
                        models_by_id[mid] = m_dict

        # 2. From model_registry table
        try:
            cur = self.lake._conn.execute("SELECT * FROM model_registry ORDER BY model_id ASC")
            for r in cur.fetchall():
                d = dict(r)
                mid = str(d.get("model_id", "")).strip().lower()
                if mid:
                    models_by_id[mid] = {
                        "model_id": d.get("model_id"),
                        "canonical_name": d.get("canonical_name", d.get("model_id")),
                        "provider": d.get("provider", "custom"),
                        "max_context_tokens": int(d.get("max_context_tokens", 0)),
                        "max_output_tokens": int(d.get("max_output_tokens", 8192)),
                        "source": d.get("source", "db"),
                    }
        except Exception:
            pass

        models = list(models_by_id.values())
        models.sort(key=lambda m: str(m.get("model_id", "")).lower())
        return models

    def export_inventory(
        self,
        include_superseded: bool = True,
        agent_filter: Optional[str] = None,
        memory_type_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export a complete, untruncated dictionary across all 5 tables in the lake.
        """
        all_nodes = self._fetch_all_nodes()
        filtered_nodes: List[Dict[str, Any]] = []

        for n in all_nodes:
            is_superseded = bool(n.get("superseded_by") or n.get("valid_until"))
            if not include_superseded and is_superseded:
                continue

            if agent_filter is not None:
                source_agent = str(n.get("source_agent", "")).strip().lower()
                if source_agent != agent_filter.strip().lower():
                    continue

            if memory_type_filter is not None:
                memory_type = str(n.get("memory_type", "")).strip().lower()
                if memory_type != memory_type_filter.strip().lower():
                    continue

            filtered_nodes.append(n)

        # Order by valid_from DESC, node_id ASC
        filtered_nodes.sort(key=lambda n: (-float(n.get("valid_from") or 0.0), str(n.get("node_id") or "")))

        edges = self._fetch_all_edges()
        sessions = self._fetch_agent_sessions()
        watermarks = self._fetch_watermarks(sessions)
        models = self._fetch_models()

        active_count = sum(1 for n in filtered_nodes if not (n.get("superseded_by") or n.get("valid_until")))
        superseded_count = sum(1 for n in filtered_nodes if bool(n.get("superseded_by") or n.get("valid_until")))
        total_tokens = sum(estimate_tokens(str(n.get("content", ""))) for n in filtered_nodes)

        summary = {
            "active_nodes": active_count,
            "superseded_nodes": superseded_count,
            "total_nodes": len(filtered_nodes),
            "total_edges": len(edges),
            "total_sessions": len(sessions),
            "total_watermarks": len(watermarks),
            "registered_models": len(models),
            "token_estimates": total_tokens,
            "total_estimated_tokens": total_tokens,
        }

        return {
            "summary": summary,
            "nodes": filtered_nodes,
            "edges": edges,
            "sessions": sessions,
            "watermarks": watermarks,
            "models": models,
        }


def list_everything(
    lake: ContextLake,
    include_superseded: bool = True,
    agent_filter: Optional[str] = None,
    memory_type_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper function to export lake inventory with default LakeInventoryExporter."""
    exporter = LakeInventoryExporter(lake)
    return exporter.export_inventory(
        include_superseded=include_superseded,
        agent_filter=agent_filter,
        memory_type_filter=memory_type_filter,
    )


def render_list_all_report(inventory: Dict[str, Any], format: str = "rich") -> str:
    """
    Render inventory in 'json', 'markdown', or 'rich' format with zero truncation of content.
    """
    fmt = (format or "rich").strip().lower()

    if fmt == "json":
        return json.dumps(inventory, indent=2, default=str)

    if fmt == "markdown":
        return _render_markdown_report(inventory)

    # Default: rich format with Console(record=True, width=120)
    return _render_rich_report(inventory)


def _render_markdown_report(inventory: Dict[str, Any]) -> str:
    """Render comprehensive zero-truncation Markdown report."""
    summary = inventory.get("summary", {})
    nodes = inventory.get("nodes", [])
    edges = inventory.get("edges", [])
    sessions = inventory.get("sessions", [])
    watermarks = inventory.get("watermarks", [])
    models = inventory.get("models", [])

    lines: List[str] = [
        "# Context Palace — Lake Inventory Report",
        "",
        "## Summary",
        f"- **Active Nodes:** {summary.get('active_nodes', 0):,}",
        f"- **Superseded Nodes:** {summary.get('superseded_nodes', 0):,}",
        f"- **Total Nodes:** {summary.get('total_nodes', 0):,}",
        f"- **Total Edges:** {summary.get('total_edges', 0):,}",
        f"- **Total Sessions:** {summary.get('total_sessions', 0):,}",
        f"- **Total Watermarks:** {summary.get('total_watermarks', 0):,}",
        f"- **Registered Models:** {summary.get('registered_models', 0):,}",
        f"- **Token Estimates:** {summary.get('token_estimates', 0):,}",
        "",
        "## Memory Nodes",
    ]

    if not nodes:
        lines.append("*(no memory nodes)*")
        lines.append("")
    else:
        lines.append("| Node ID | Memory Type | Tier | Source Agent | Confidence | Valid From | Valid Until | Superseded By |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for n in nodes:
            v_until = n.get("valid_until") if n.get("valid_until") is not None else "None"
            sup_by = n.get("superseded_by") if n.get("superseded_by") is not None else "None"
            conf = f"{float(n.get('confidence', 0.0)):.2f}"
            lines.append(
                f"| {n.get('node_id', '')} | {n.get('memory_type', '')} | {n.get('tier', '')} | "
                f"{n.get('source_agent', '')} | {conf} | {n.get('valid_from', '')} | {v_until} | {sup_by} |"
            )
        lines.append("")
        lines.append("### Node Content Details")
        lines.append("")
        for n in nodes:
            v_until = n.get("valid_until") if n.get("valid_until") is not None else "None"
            sup_by = n.get("superseded_by") if n.get("superseded_by") is not None else "None"
            lines.append(f"#### `{n.get('node_id', '')}`")
            lines.append(f"- **Memory Type:** {n.get('memory_type', '')}")
            lines.append(f"- **Tier:** {n.get('tier', '')}")
            lines.append(f"- **Source Agent:** {n.get('source_agent', '')}")
            lines.append(f"- **Confidence:** {float(n.get('confidence', 0.0)):.2f}")
            lines.append(f"- **Valid From:** {n.get('valid_from', '')}")
            lines.append(f"- **Valid Until:** {v_until}")
            lines.append(f"- **Superseded By:** {sup_by}")
            lines.append("- **Content:**")
            lines.append("```")
            lines.append(str(n.get("content", "")))
            lines.append("```")
            lines.append("")

    lines.append("## Edges")
    if not edges:
        lines.append("*(no edges)*")
        lines.append("")
    else:
        lines.append("| Edge ID | Source ID | Target ID | Relation Type | Weight | Source Agent | Valid From | Valid Until |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for e in edges:
            v_until = e.get("valid_until") if e.get("valid_until") is not None else "None"
            lines.append(
                f"| {e.get('edge_id', '')} | {e.get('source_id', '')} | {e.get('target_id', '')} | "
                f"{e.get('relation_type', '')} | {float(e.get('weight', 1.0)):.2f} | {e.get('source_agent', '')} | "
                f"{e.get('valid_from', '')} | {v_until} |"
            )
        lines.append("")

    lines.append("## Agent Sessions")
    if not sessions:
        lines.append("*(no agent sessions)*")
        lines.append("")
    else:
        lines.append("| Agent ID | Harness | Model | Used Tokens | Max Context | Last Turn | Last Offset | Updated At |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for s in sessions:
            lines.append(
                f"| {s.get('agent_id', '')} | {s.get('harness_type', '')} | {s.get('model_id', '')} | "
                f"{int(s.get('used_tokens', 0)):,} | {int(s.get('max_context_tokens', 0)):,} | "
                f"{s.get('last_processed_turn', 0)} | {s.get('last_byte_offset', 0)} | {s.get('updated_at', 0.0)} |"
            )
        lines.append("")

    lines.append("## Extraction Watermarks")
    if not watermarks:
        lines.append("*(no extraction watermarks)*")
        lines.append("")
    else:
        lines.append("| Agent ID | Transcript Path | Last Turn | Last Offset | Updated At |")
        lines.append("| --- | --- | --- | --- | --- |")
        for w in watermarks:
            lines.append(
                f"| {w.get('agent_id', '')} | {w.get('transcript_path') or 'None'} | "
                f"{w.get('last_turn', 0)} | {w.get('last_offset', 0)} | {w.get('updated_at', 0.0)} |"
            )
        lines.append("")

    lines.append("## Model Registry")
    if not models:
        lines.append("*(no models registered)*")
        lines.append("")
    else:
        lines.append("| Model ID | Canonical Name | Provider | Max Context Tokens | Max Output Tokens | Source |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for m in models:
            lines.append(
                f"| {m.get('model_id', '')} | {m.get('canonical_name', '')} | {m.get('provider', '')} | "
                f"{int(m.get('max_context_tokens', 0)):,} | {int(m.get('max_output_tokens', 0)):,} | {m.get('source', '')} |"
            )
        lines.append("")

    return "\n".join(lines)


def _render_rich_report(inventory: Dict[str, Any]) -> str:
    """Render comprehensive zero-truncation Rich console report."""
    console = Console(record=True, width=120)
    summary = inventory.get("summary", {})
    nodes = inventory.get("nodes", [])
    edges = inventory.get("edges", [])
    sessions = inventory.get("sessions", [])
    watermarks = inventory.get("watermarks", [])
    models = inventory.get("models", [])

    # 1. Summary Panel
    summary_table = Table(title="Lake Inventory Summary", expand=True, show_lines=False)
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Count / Value", style="bold white", justify="right")
    summary_table.add_row("Active Memory Nodes", f"{summary.get('active_nodes', 0):,}")
    summary_table.add_row("Superseded Memory Nodes", f"{summary.get('superseded_nodes', 0):,}")
    summary_table.add_row("Total Memory Nodes", f"{summary.get('total_nodes', 0):,}")
    summary_table.add_row("Total Memory Edges", f"{summary.get('total_edges', 0):,}")
    summary_table.add_row("Total Agent Sessions", f"{summary.get('total_sessions', 0):,}")
    summary_table.add_row("Total Extraction Watermarks", f"{summary.get('total_watermarks', 0):,}")
    summary_table.add_row("Registered Models", f"{summary.get('registered_models', 0):,}")
    summary_table.add_row("Estimated Node Content Tokens", f"{summary.get('token_estimates', 0):,}")

    console.print(Panel(summary_table, title="Context Palace — Lake Inventory", border_style="bright_blue"))

    # 2. Table: memory_nodes
    nodes_table = Table(title=f"Memory Nodes ({len(nodes):,} rows — Zero Truncation)", expand=True, show_lines=True)
    nodes_table.add_column("Node ID", style="bold cyan", no_wrap=True)
    nodes_table.add_column("Type", style="magenta")
    nodes_table.add_column("Tier", style="green")
    nodes_table.add_column("Source Agent", style="yellow")
    nodes_table.add_column("Conf", justify="right")
    nodes_table.add_column("Valid From", style="dim")
    nodes_table.add_column("Valid Until", style="dim")
    nodes_table.add_column("Superseded By", style="red")
    nodes_table.add_column("Content (Untruncated)", style="white")

    if not nodes:
        nodes_table.add_row("-", "(empty)", "-", "-", "-", "-", "-", "-", "(no memory nodes)")
    else:
        for n in nodes:
            v_until = str(n.get("valid_until")) if n.get("valid_until") is not None else "-"
            sup_by = str(n.get("superseded_by")) if n.get("superseded_by") is not None else "-"
            nodes_table.add_row(
                str(n.get("node_id", "")),
                str(n.get("memory_type", "")),
                str(n.get("tier", "")),
                str(n.get("source_agent", "")),
                f"{float(n.get('confidence', 0.0)):.2f}",
                str(n.get("valid_from", "")),
                v_until,
                sup_by,
                str(n.get("content", "")),  # NEVER TRUNCATED!
            )
    console.print(nodes_table)

    # 3. Table: memory_edges
    edges_table = Table(title=f"Memory Edges ({len(edges):,} rows)", expand=True)
    edges_table.add_column("Edge ID", style="bold cyan")
    edges_table.add_column("Source ID", style="white")
    edges_table.add_column("Target ID", style="white")
    edges_table.add_column("Relation Type", style="yellow")
    edges_table.add_column("Weight", justify="right")
    edges_table.add_column("Source Agent", style="magenta")
    edges_table.add_column("Valid From", style="dim")
    edges_table.add_column("Valid Until", style="dim")

    if not edges:
        edges_table.add_row("-", "-", "-", "(empty)", "-", "-", "-", "-")
    else:
        for e in edges:
            v_until = str(e.get("valid_until")) if e.get("valid_until") is not None else "-"
            edges_table.add_row(
                str(e.get("edge_id", "")),
                str(e.get("source_id", "")),
                str(e.get("target_id", "")),
                str(e.get("relation_type", "")),
                f"{float(e.get('weight', 1.0)):.2f}",
                str(e.get("source_agent", "")),
                str(e.get("valid_from", "")),
                v_until,
            )
    console.print(edges_table)

    # 4. Table: agent_sessions
    sessions_table = Table(title=f"Agent Sessions ({len(sessions):,} rows)", expand=True)
    sessions_table.add_column("Agent ID", style="bold cyan")
    sessions_table.add_column("Harness", style="magenta")
    sessions_table.add_column("Model ID", style="green")
    sessions_table.add_column("Used / Max Tokens", justify="right")
    sessions_table.add_column("Last Turn", justify="right")
    sessions_table.add_column("Last Offset", justify="right")
    sessions_table.add_column("Updated At", style="dim")

    if not sessions:
        sessions_table.add_row("-", "(empty)", "-", "-", "-", "-", "-")
    else:
        for s in sessions:
            used = int(s.get("used_tokens", 0))
            max_t = int(s.get("max_context_tokens", 0))
            sessions_table.add_row(
                str(s.get("agent_id", "")),
                str(s.get("harness_type", "")),
                str(s.get("model_id", "")),
                f"{used:,} / {max_t:,}",
                str(s.get("last_processed_turn", 0)),
                str(s.get("last_byte_offset", 0)),
                str(s.get("updated_at", 0.0)),
            )
    console.print(sessions_table)

    # 5. Table: extraction_watermarks
    wm_table = Table(title=f"Extraction Watermarks ({len(watermarks):,} rows)", expand=True)
    wm_table.add_column("Agent ID", style="bold cyan")
    wm_table.add_column("Transcript Path", style="white")
    wm_table.add_column("Last Turn", justify="right")
    wm_table.add_column("Last Offset", justify="right")
    wm_table.add_column("Updated At", style="dim")

    if not watermarks:
        wm_table.add_row("-", "(empty)", "-", "-", "-")
    else:
        for w in watermarks:
            wm_table.add_row(
                str(w.get("agent_id", "")),
                str(w.get("transcript_path") or "-"),
                str(w.get("last_turn", 0)),
                str(w.get("last_offset", 0)),
                str(w.get("updated_at", 0.0)),
            )
    console.print(wm_table)

    # 6. Table: model_registry
    models_table = Table(title=f"Model Registry ({len(models):,} rows)", expand=True)
    models_table.add_column("Model ID", style="bold cyan")
    models_table.add_column("Canonical Name", style="white")
    models_table.add_column("Provider", style="magenta")
    models_table.add_column("Max Context", justify="right")
    models_table.add_column("Max Output", justify="right")
    models_table.add_column("Source", style="green")

    if not models:
        models_table.add_row("-", "(empty)", "-", "-", "-", "-")
    else:
        for m in models:
            models_table.add_row(
                str(m.get("model_id", "")),
                str(m.get("canonical_name", "")),
                str(m.get("provider", "")),
                f"{int(m.get('max_context_tokens', 0)):,}",
                f"{int(m.get('max_output_tokens', 0)):,}",
                str(m.get("source", "")),
            )
    console.print(models_table)

    return console.export_text()
