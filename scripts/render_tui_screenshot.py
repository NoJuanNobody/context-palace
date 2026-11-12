"""
Render the Textual TUI against a seeded demo lake and save it as SVG (and PNG if cairosvg
is installed). Synthetic data only; never points at ~/.context_palace.

    python scripts/render_tui_screenshot.py out/tui.svg [--png out/tui.png] [--cols 200 --rows 52]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from context_palace.cli import _load_components  # noqa: E402
from context_palace.tui_app import ContextPalaceApp  # noqa: E402
from seed_demo_lake import seed  # noqa: E402


async def _render(db: str, cols: int, rows: int, push_capsule: bool, chat: str = "", list_all: bool = False) -> str:
    comps = _load_components(db)
    tui = comps["tui"]
    if push_capsule:
        tui.execute_key_action("p", {"node_id_or_name": "auth_strategy", "agent_id": "claude-code-dev",
                                     "query": "auth", "sidecar_path": os.path.join(os.path.dirname(db), ".context_capsule.md")})
        tui.last_output = []
        tui.status_message = "Ready"
    if chat:
        tui.handle_command(f"a {chat}")
        tui.status_message = "Ready"
    app = ContextPalaceApp(tui, refresh_seconds=0)
    async with app.run_test(size=(cols, rows)) as pilot:
        await pilot.pause()
        if list_all:
            await pilot.press("l")
            await pilot.pause()
        return app.export_screenshot(title="Context Palace — context-palace tui")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svg")
    ap.add_argument("--png", default=None)
    ap.add_argument("--cols", type=int, default=200)
    ap.add_argument("--rows", type=int, default=52)
    ap.add_argument("--no-capsule", action="store_true")
    ap.add_argument("--chat", default="What do my agents know about auth_strategy?",
                    help="[a] Ask turn to show in AUDIT ('' for none)")
    ap.add_argument("--list-all", action="store_true", help="capture the [l] List All inventory view")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "context_lake.db")
        seed(db)
        svg = asyncio.run(_render(db, args.cols, args.rows, not args.no_capsule, args.chat, args.list_all))
    os.makedirs(os.path.dirname(os.path.abspath(args.svg)), exist_ok=True)
    with open(args.svg, "w", encoding="utf-8") as fh:
        fh.write(svg)
    if args.png:
        import cairosvg  # type: ignore

        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=args.png, output_width=args.cols * 12)
    print(args.svg, args.png or "")


if __name__ == "__main__":
    main()
