"""Serve the power-grid visualization HTML files over local HTTP."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from power.asset_registry_viewer import (
    DEFAULT_VIEWER_PORT,
    POWER_GRID,
    viewer_urls,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the Marshfield power-grid HTML viewers."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_VIEWER_PORT)
    args = parser.parse_args()

    handler = partial(SimpleHTTPRequestHandler, directory=str(POWER_GRID))
    urls = viewer_urls(args.host, args.port)

    print("Serving Marshfield sandbox visualizations")
    print(f"Directory: {POWER_GRID}")
    print(f"Recommended Asset Registry viewer: {urls['recommended']}")
    print(f"Optional deck.gl Asset Registry:   {urls['deckgl']}")
    print(f"Legacy OpenDSS infrastructure map: {urls['grid_map']}")
    print("Press Ctrl-C to stop.")

    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
