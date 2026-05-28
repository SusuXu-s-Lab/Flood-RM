"""Entry point for fragility-based flood impact analysis."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import sys

from power.asset_registry_viewer import POWER_GRID
from power import impact_analysis


DEFAULT_IMPACT_VIEWER_PORT = 8766


def serve_viewer(host: str, port: int, viewer_path) -> None:
    url = impact_analysis.impact_viewer_url(host, port, viewer_path)
    handler = partial(SimpleHTTPRequestHandler, directory=str(POWER_GRID))
    print(f"Impact viewer: {url}", flush=True)
    print(f"Serving Marshfield impact viewer from {POWER_GRID}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    with ThreadingHTTPServer((host, port), handler) as server:
        server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    command = "compute"
    if raw_args[:1] in (["compute"], ["serve"]):
        command = raw_args[0]
        raw_args = raw_args[1:]

    parser = argparse.ArgumentParser(
        description="Compute Marshfield power-grid flood impacts and launch an HTML impact map."
    )
    impact_analysis.add_impact_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_IMPACT_VIEWER_PORT)
    parser.add_argument("--no-serve", action="store_true", help="write outputs and viewer HTML without starting a local server")
    args = parser.parse_args(raw_args)

    if command == "serve":
        try:
            serve_viewer(args.host, args.port, args.viewer_output)
        except KeyboardInterrupt:
            return 0
        return 0

    event_id = args.event_id or args.event_dir.name
    output_dir = args.output_dir or impact_analysis.IMPACTS_DIR / event_id
    rows, summary = impact_analysis.compute_asset_impacts(
        args.event_dir,
        event_id=event_id,
        probability_threshold=args.probability_threshold,
        max_sample_distance_m=args.max_sample_distance_m,
        include_lines=args.include_lines,
    )
    impact_analysis.write_outputs(rows, summary, output_dir)
    viewer_path = impact_analysis.write_impact_viewer(
        rows,
        summary,
        args.viewer_output,
        event_dir=args.event_dir,
        flood_spatial_stride=args.flood_spatial_stride,
        flood_time_stride=args.flood_time_stride,
    )
    url = impact_analysis.impact_viewer_url(args.host, args.port, viewer_path)

    print(f"Wrote {len(rows):,} asset impacts to {output_dir / 'asset_impacts.csv'}", flush=True)
    print(f"Wrote impact map to {viewer_path}", flush=True)
    print(f"Expected affected assets: {summary['expected_affected_count']:.2f}", flush=True)
    print(f"Affected assets at p >= {args.probability_threshold:g}: {summary['affected_count']:,}", flush=True)

    if args.no_serve:
        print("Not serving because --no-serve was set.", flush=True)
        print(f"Start it later with: uv run impact_viewer.py serve --port {args.port}", flush=True)
        print(f"Viewer path: {viewer_path}", flush=True)
        return 0

    try:
        serve_viewer(args.host, args.port, viewer_path)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
