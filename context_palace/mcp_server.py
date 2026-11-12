"""
MCP (Model Context Protocol) JSON-RPC Bridge for the Federated Context Lake (context-palace).
Exposes 6 MCP tools for proactive subgraph queries, model-window-adaptive reactive capsules,
concept CRUD, soft/hard deletion, bitemporal time-travel snapshots, and live context window telemetry.
"""

from __future__ import annotations

import inspect
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from context_palace.models import (
    AgentSession,
    ConceptNode,
    RelationEdge,
    compute_text_embedding,
)
from context_palace.retrieval import DualDirectionRetriever, _coerce_edge, _coerce_node


class ContextPalaceMCPServer:
    """
    MCP Server exposing the Federated Context Lake to Antigravity, Claude Code, Codex,
    and local Ollama coding agents.
    """

    def __init__(
        self,
        lake: Any,
        retriever: Optional[DualDirectionRetriever] = None,
        registry: Optional[Any] = None,
    ) -> None:
        self.lake = lake
        self.retriever = retriever or DualDirectionRetriever(lake)
        self.registry = registry

    def list_tools(self, include_v2: bool = False) -> List[Dict[str, Any]]:
        """Return MCP tool definitions with JSON schemas for Context Palace tools."""
        tools = [
            {
                "name": "context_query",
                "description": (
                    "Proactive pull subgraph query across the Federated Context Lake. "
                    "Scores concepts by vector similarity, keyword match, confidence, and pinned priority, "
                    "then expands multi-hop edges."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query or task description to match against concepts.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of seed concepts to retrieve before hop expansion.",
                            "default": 8,
                        },
                        "hops": {
                            "type": "integer",
                            "description": "Number of graph hops to expand along incident edges.",
                            "default": 1,
                        },
                        "as_of": {
                            "type": ["number", "null"],
                            "description": "Optional historical Unix timestamp for bitemporal point-in-time query.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "context_inject_capsule",
                "description": (
                    "Reactive push capsule builder fitted to an agent's auto-detected model context window headroom. "
                    "Optionally writes the capsule idempotently into a sidecar Markdown file."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Target agent session identifier.",
                        },
                        "task_query": {
                            "type": "string",
                            "description": "Current task or focus query for prioritizing relevant concepts.",
                            "default": "",
                        },
                        "budget_tokens": {
                            "type": ["integer", "null"],
                            "description": "Optional explicit token budget override; defaults to adaptive model headroom budget.",
                        },
                        "sidecar_path": {
                            "type": ["string", "null"],
                            "description": "Optional file path (e.g. .context_capsule.md or AGENTS.md) to inject the capsule into.",
                        },
                    },
                    "required": ["agent_id"],
                },
            },
            {
                "name": "context_upsert",
                "description": "Create or update a concept in the Federated Context Lake.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "canonical_name": {
                            "type": "string",
                            "description": "Canonical unique name of the concept or architectural fact.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Concept category (e.g. architecture, config, constraint, entity, decision, task).",
                            "default": "architecture",
                        },
                        "value": {
                            "type": "string",
                            "description": "Active belief value or specification for the concept.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Optional short summary of the concept.",
                            "default": "",
                        },
                        "pinned": {
                            "type": "boolean",
                            "description": "Whether this concept should be pinned into all reactive Context Capsules.",
                            "default": False,
                        },
                        "source_agent_id": {
                            "type": "string",
                            "description": "Identifier of the agent or operator asserting this belief.",
                            "default": "mcp_client",
                        },
                    },
                    "required": ["canonical_name", "value"],
                },
            },
            {
                "name": "context_forget",
                "description": "Soft-tombstone or hard-delete a concept from the Federated Context Lake.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node_id_or_name": {
                            "type": "string",
                            "description": "Node ID or canonical name of the concept to forget.",
                        },
                        "hard": {
                            "type": "boolean",
                            "description": "If true, permanently delete; if false (default), soft-tombstone.",
                            "default": False,
                        },
                    },
                    "required": ["node_id_or_name"],
                },
            },
            {
                "name": "context_time_travel",
                "description": "Reconstruct the belief graph at a historical Unix timestamp.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "as_of": {
                            "type": "number",
                            "description": "Historical Unix timestamp (seconds) at which to reconstruct the belief graph.",
                        },
                    },
                    "required": ["as_of"],
                },
            },
            {
                "name": "context_window_status",
                "description": (
                    "Return auto-detected model context window capacities, token usage, headroom, "
                    "and compaction alerts for connected agents."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": ["string", "null"],
                            "description": "Optional agent ID to filter status for a single connected session.",
                        },
                    },
                },
            },
        ]
        if include_v2:
            tools.extend(
                [
                    {
                        "name": "context_list_all",
                        "description": (
                            "Return the complete untruncated inventory across all 5 tables: "
                            "memory_nodes, memory_edges, agent_sessions, extraction_watermarks, model_registry."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "active_only": {
                                    "type": "boolean",
                                    "description": "If true, exclude tombstoned nodes and inactive edges.",
                                    "default": False,
                                },
                                "agent_id": {
                                    "type": ["string", "null"],
                                    "description": "Optional filter by source agent ID.",
                                },
                                "category": {
                                    "type": ["string", "null"],
                                    "description": "Optional filter by concept category.",
                                },
                            },
                        },
                    },
                    {
                        "name": "context_chat",
                        "description": (
                            "Execute a natural-language conversational query or memory command against the "
                            "Federated Context Lake and return the grounded answer, intent, and cited nodes."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "Natural-language query, question, or CRUD instruction.",
                                },
                                "as_of": {
                                    "type": ["number", "null"],
                                    "description": "Optional historical Unix timestamp for bitemporal point-in-time query.",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Maximum number of seed concepts to retrieve.",
                                    "default": 8,
                                },
                            },
                            "required": ["message"],
                        },
                    },
                ]
            )
        return tools

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP tool definitions with JSON schemas for all Context Palace tools including v2."""
        return self.list_tools(include_v2=True)

    def _dispatch_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch an MCP tool execution (alias for call_tool)."""
        return self.call_tool(name, arguments)

    def _resolve_session(self, agent_id: str) -> AgentSession:
        # 1. Try lake.get_session / get_agent_session
        for target in (self.lake, self.registry):
            if target is None:
                continue
            for method_name in ("get_session", "get_agent_session"):
                if hasattr(target, method_name):
                    sess = getattr(target, method_name)(agent_id)
                    if isinstance(sess, AgentSession):
                        return sess
                    if isinstance(sess, dict):
                        return AgentSession(
                            agent_id=str(sess.get("agent_id", agent_id)),
                            harness_type=str(sess.get("harness_type", "antigravity")),
                            model_id=str(sess.get("model_id", "gemini-2.5-pro")),
                            max_context_tokens=int(sess.get("max_context_tokens", 1_048_576)),
                            used_tokens=int(sess.get("used_tokens", 0)),
                            transcript_path=sess.get("transcript_path"),
                            last_processed_turn=int(sess.get("last_processed_turn", 0)),
                            last_byte_offset=int(sess.get("last_byte_offset", 0)),
                            updated_at=float(sess.get("updated_at", time.time())),
                            metadata=dict(sess.get("metadata", {})),
                        )
            for list_name in ("list_sessions", "list_agent_sessions"):
                if hasattr(target, list_name):
                    for s in getattr(target, list_name)():
                        s_id = s.agent_id if isinstance(s, AgentSession) else s.get("agent_id")
                        if s_id == agent_id:
                            if isinstance(s, AgentSession):
                                return s
                            return AgentSession(
                                agent_id=str(s.get("agent_id", agent_id)),
                                harness_type=str(s.get("harness_type", "antigravity")),
                                model_id=str(s.get("model_id", "gemini-2.5-pro")),
                                max_context_tokens=int(s.get("max_context_tokens", 1_048_576)),
                                used_tokens=int(s.get("used_tokens", 0)),
                            )

        # Fallback default session (auto-resolved via registry if available)
        max_ctx = 200_000
        model_id = "claude-3-7-sonnet"
        if self.registry is not None and hasattr(self.registry, "resolve"):
            spec = self.registry.resolve(model_id)
            max_ctx = getattr(spec, "max_context_tokens", max_ctx)
        return AgentSession(
            agent_id=agent_id,
            harness_type="custom",
            model_id=model_id,
            max_context_tokens=max_ctx,
            used_tokens=24_000,
            updated_at=time.time(),
        )

    def _list_all_sessions(self, agent_id: Optional[str] = None) -> List[AgentSession]:
        sessions: List[AgentSession] = []
        seen: set[str] = set()
        for target in (self.lake, self.registry):
            if target is None:
                continue
            for list_name in ("list_sessions", "list_agent_sessions"):
                if hasattr(target, list_name):
                    for s in getattr(target, list_name)():
                        sess = s if isinstance(s, AgentSession) else self._resolve_session(s.get("agent_id", "default"))
                        if sess.agent_id not in seen:
                            seen.add(sess.agent_id)
                            sessions.append(sess)
        if agent_id:
            filtered = [s for s in sessions if s.agent_id == agent_id]
            if not filtered:
                filtered = [self._resolve_session(agent_id)]
            return filtered
        return sessions

    def _invoke_upsert(self, args: Dict[str, Any]) -> ConceptNode:
        canonical_name = str(args["canonical_name"])
        category = str(args.get("category", "architecture"))
        value = str(args["value"])
        summary = str(args.get("summary", ""))
        pinned = bool(args.get("pinned", False))
        source_agent_id = str(args.get("source_agent_id", "mcp_client"))
        confidence = float(args.get("confidence", 0.95))

        if hasattr(self.lake, "upsert_concept"):
            fn = self.lake.upsert_concept
            sig = inspect.signature(fn)
            params = sig.parameters
            call_kwargs: Dict[str, Any] = {}
            mapping = {
                "canonical_name": canonical_name,
                "name": canonical_name,
                "category": category,
                "value": value,
                "active_value": value,
                "summary": summary,
                "pinned": pinned,
                "source_agent_id": source_agent_id,
                "confidence": confidence,
                "provenance_type": str(args.get("provenance_type", "operator_crud")),
            }
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            for k, v in mapping.items():
                if k in params:
                    call_kwargs[k] = v
            if accepts_var_kw and not call_kwargs:
                call_kwargs = {
                    "canonical_name": canonical_name,
                    "category": category,
                    "value": value,
                    "summary": summary,
                    "pinned": pinned,
                    "source_agent_id": source_agent_id,
                    "confidence": confidence,
                }
            res = fn(**call_kwargs)
            if isinstance(res, tuple):
                res = res[0]
            return _coerce_node(res)

        now = time.time()
        node = ConceptNode(
            node_id=f"node_{uuid.uuid4().hex[:10]}",
            canonical_name=canonical_name,
            category=category,
            active_value=value,
            summary=summary,
            confidence=confidence,
            pinned=pinned,
            embedding=compute_text_embedding(f"{canonical_name} {category} {value} {summary}"),
            source_agent_id=source_agent_id,
            created_at=now,
            updated_at=now,
            tombstoned=False,
        )
        return node

    def _invoke_forget(self, node_id_or_name: str, hard: bool = False) -> bool:
        for method_name in ("forget_concept", "delete_concept", "remove_concept"):
            if hasattr(self.lake, method_name):
                fn = getattr(self.lake, method_name)
                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())
                try:
                    if "hard" in params:
                        return bool(fn(node_id_or_name, hard=hard))
                    return bool(fn(node_id_or_name))
                except TypeError:
                    return bool(fn(node_id_or_name))
        return False

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one of the 6 Context Palace MCP tools and return its structured result."""
        args = arguments or {}

        if name == "context_query":
            query = str(args.get("query", ""))
            top_k = int(args.get("top_k", 8))
            hops = int(args.get("hops", 1))
            as_of = float(args["as_of"]) if args.get("as_of") is not None else None
            nodes, edges = self.retriever.query_subgraph(
                query=query,
                top_k=top_k,
                hops=hops,
                as_of=as_of,
                include_pinned= bool(args.get("include_pinned", True)),
            )
            return {
                "query": query,
                "top_k": top_k,
                "hops": hops,
                "as_of": as_of,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "nodes": [n.to_dict() for n in nodes],
                "edges": [e.to_dict() for e in edges],
            }

        if name == "context_inject_capsule":
            agent_id = str(args.get("agent_id", "default_agent"))
            task_query = str(args.get("task_query", ""))
            budget_tokens = (
                int(args["budget_tokens"]) if args.get("budget_tokens") is not None else None
            )
            sidecar_path = args.get("sidecar_path")
            as_of = float(args["as_of"]) if args.get("as_of") is not None else None

            session = self._resolve_session(agent_id)
            capsule = self.retriever.build_context_capsule(
                session=session,
                task_query=task_query,
                budget_tokens=budget_tokens,
                as_of=as_of,
            )
            result = capsule.to_dict()
            if sidecar_path:
                self.retriever.inject_into_sidecar(capsule, str(sidecar_path))
                result["sidecar_path"] = str(sidecar_path)
                result["sidecar_injected"] = True
            else:
                result["sidecar_injected"] = False
            return result

        if name == "context_upsert":
            node = self._invoke_upsert(args)
            return {
                "status": "upserted",
                "node": node.to_dict(),
            }

        if name == "context_forget":
            target = str(args["node_id_or_name"])
            hard = bool(args.get("hard", False))
            forgotten = self._invoke_forget(target, hard=hard)
            return {
                "status": "deleted" if hard else "tombstoned",
                "node_id_or_name": target,
                "hard": hard,
                "success": forgotten,
            }

        if name == "context_time_travel":
            as_of = float(args["as_of"])
            nodes, edges = self.retriever._load_graph(as_of=as_of)
            return {
                "as_of": as_of,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "nodes": [n.to_dict() for n in nodes],
                "edges": [e.to_dict() for e in edges],
            }

        if name == "context_window_status":
            agent_id = args.get("agent_id")
            sessions = self._list_all_sessions(agent_id=str(agent_id) if agent_id else None)
            session_dicts = []
            compaction_alerts = []
            for s in sessions:
                d = s.to_dict()
                d["adaptive_capsule_budget"] = s.adaptive_capsule_budget()
                session_dicts.append(d)
                if s.needs_compaction:
                    compaction_alerts.append(
                        {
                            "agent_id": s.agent_id,
                            "model_id": s.model_id,
                            "utilization_pct": s.utilization_pct,
                            "remaining_headroom": s.remaining_headroom,
                            "recommended_capsule_budget": s.adaptive_capsule_budget(),
                        }
                    )
            return {
                "session_count": len(session_dicts),
                "sessions": session_dicts,
                "compaction_alerts": compaction_alerts,
            }

        if name == "context_list_all":
            active_only = bool(args.get("active_only", False))
            agent_id = args.get("agent_id")
            category = args.get("category")

            nodes = (
                self.lake.list_concepts(include_tombstoned=not active_only)
                if hasattr(self.lake, "list_concepts")
                else []
            )
            if category:
                nodes = [n for n in nodes if getattr(n, "category", "") == category]
            if agent_id:
                nodes = [n for n in nodes if getattr(n, "source_agent_id", "") == agent_id]

            edges = (
                self.lake.list_edges(active_only=active_only)
                if hasattr(self.lake, "list_edges")
                else []
            )
            if agent_id:
                edges = [e for e in edges if getattr(e, "source_agent_id", "") == agent_id]

            sessions = (
                self.lake.list_agent_sessions()
                if hasattr(self.lake, "list_agent_sessions")
                else []
            )
            if agent_id:
                sessions = [s for s in sessions if getattr(s, "agent_id", "") == agent_id]

            watermarks = [
                {
                    "agent_id": s.agent_id,
                    "last_byte_offset": s.last_byte_offset,
                    "last_processed_turn": s.last_processed_turn,
                    "transcript_path": s.transcript_path,
                    "updated_at": s.updated_at,
                }
                for s in sessions
            ]

            models = [
                {
                    "model_id": s.model_id,
                    "harness_type": s.harness_type,
                    "max_context_tokens": s.max_context_tokens,
                    "used_tokens": s.used_tokens,
                }
                for s in sessions
            ]

            return {
                "status": "ok",
                "memory_nodes": [n.to_dict() if hasattr(n, "to_dict") else n for n in nodes],
                "memory_edges": [e.to_dict() if hasattr(e, "to_dict") else e for e in edges],
                "agent_sessions": [s.to_dict() if hasattr(s, "to_dict") else s for s in sessions],
                "extraction_watermarks": watermarks,
                "model_registry": models,
                "counts": {
                    "memory_nodes": len(nodes),
                    "memory_edges": len(edges),
                    "agent_sessions": len(sessions),
                    "extraction_watermarks": len(watermarks),
                    "model_registry": len(models),
                },
            }

        if name == "context_chat":
            message = str(args.get("message", ""))
            as_of = float(args["as_of"]) if args.get("as_of") is not None else None
            top_k = int(args.get("top_k", 8))

            answer = ""
            cited_nodes: List[str] = []
            intent = "query"

            try:
                from context_palace.conversation import ConversationalMemoryPalace  # type: ignore
                conv = ConversationalMemoryPalace(
                    lake=self.lake, retriever=self.retriever, registry=self.registry
                )
                turn = conv.ask(message, as_of=as_of, top_k=top_k)
                answer = getattr(turn, "answer", str(turn.get("answer", "")))
                cited_nodes = getattr(turn, "cited_nodes", list(turn.get("cited_nodes", [])))
                intent = getattr(turn, "intent", str(turn.get("intent", "query")))
            except Exception:
                from context_palace.tui import _synthesize_chat_turn
                turn = _synthesize_chat_turn(
                    self.lake, self.retriever, message, as_of=as_of, top_k=top_k
                )
                answer = turn["answer"]
                cited_nodes = turn["cited_nodes"]
                intent = turn["intent"]

            return {
                "status": "ok",
                "message": message,
                "answer": answer,
                "intent": intent,
                "cited_nodes": cited_nodes,
                "as_of": as_of,
            }

        raise ValueError(f"Unknown MCP tool: {name}")

    def handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard MCP JSON-RPC 2.0 requests (`initialize`, `tools/list`, `tools/call`)."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {
                            "name": "context-palace-mcp",
                            "version": "0.1.0",
                        },
                        "capabilities": {
                            "tools": {"listChanged": False},
                        },
                    },
                }

            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": self.list_tools(),
                    },
                }

            if method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments") or {}
                payload = self.call_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(payload, indent=2),
                            }
                        ],
                        "structuredContent": payload,
                        "isError": False,
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": str(exc),
                },
            }
