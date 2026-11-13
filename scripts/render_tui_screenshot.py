"""
Render the Textual TUI against a Context Lake DB and save it as SVG (and PNG if cairosvg
is installed).

    python scripts/render_tui_screenshot.py out/tui.svg [--db ~/.context_palace/context_lake.db] [--png out/tui.png] [--cols 200 --rows 52]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from context_palace.cli import _load_components  # noqa: E402
from context_palace.tui_app import ContextPalaceApp  # noqa: E402


async def _render(db: str, cols: int, rows: int, chat: str = "", list_all: bool = False) -> str:
    comps = _load_components(db)
    tui = comps["tui"]
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
    ap.add_argument("--db", default=None, help="Path to context_lake.db (defaults to an empty temporary DB)")
    ap.add_argument("--png", default=None)
    ap.add_argument("--cols", type=int, default=200)
    ap.add_argument("--rows", type=int, default=52)
    ap.add_argument("--chat", default="",
                    help="[a] Ask turn to show in AUDIT ('' for none)")
    ap.add_argument("--list-all", action="store_true", help="capture the [l] List All inventory view")
    args = ap.parse_args()
    if args.db:
        svg = asyncio.run(_render(os.path.expanduser(args.db), args.cols, args.rows, args.chat, args.list_all))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "context_lake.db")
            svg = asyncio.run(_render(db, args.cols, args.rows, args.chat, args.list_all))
    os.makedirs(os.path.dirname(os.path.abspath(args.svg)), exist_ok=True)
    with open(args.svg, "w", encoding="utf-8") as fh:
        fh.write(svg)
    if args.png:
        import cairosvg  # type: ignore

        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=args.png, output_width=args.cols * 12)
    print(args.svg, args.png or "")


if __name__ == "__main__":
    main()
