"""Zero-dependency JSON API server for the engine UI.

Runs on the Python standard library only (``http.server``) — no Flask/FastAPI —
to keep the engine's install footprint unchanged.  It exposes:

    GET /api/health                       -> {"status": "ok", ...}
    GET /api/indices                      -> available indices for the loaded data
    GET /api/report?index=NDX&objective=growth&risk=4&drift=0.08
                                          -> full engine report (JSON)

and, if a built UI is present at ``ui/dist``, serves it statically so the whole
app runs from one process.  Start it with::

    python -m wharton_ml_engine.api.server --port 8000

The React dev server (``npm run dev`` in ``ui/``) proxies ``/api`` here, so you
can develop the UI live against a running engine.
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .service import EngineService

_SERVICE: Optional[EngineService] = None
_UI_DIST = os.path.join("ui", "dist")

_MIME = {
    ".html": "text/html", ".js": "application/javascript", ".css": "text/css",
    ".json": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon",
    ".map": "application/json", ".woff2": "font/woff2",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # quieter logs
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        with open(path, "rb") as fh:
            body = fh.read()
        ext = os.path.splitext(path)[1]
        self.send_response(200)
        self.send_header("Content-Type", _MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)

        try:
            if path == "/api/health":
                return self._send_json({
                    "status": "ok",
                    "dataset": os.path.basename(_SERVICE.dataset_dir),
                    "has_model": _SERVICE.alpha_model is not None,
                })
            if path == "/api/indices":
                return self._send_json(_SERVICE.indices())
            if path == "/api/report":
                drift = q.get("drift", [None])[0]
                out = _SERVICE.report(
                    index_id=q.get("index", ["SPX"])[0],
                    objective=q.get("objective", ["balanced"])[0],
                    risk_tolerance=int(q.get("risk", ["3"])[0]),
                    mc_sims=int(q.get("sims", ["8000"])[0]),
                    mc_annual_drift=float(drift) if drift not in (None, "") else None,
                )
                return self._send_json(out)
        except (ValueError, KeyError) as e:
            return self._send_json({"error": str(e)}, status=400)
        except Exception as e:                              # pragma: no cover
            return self._send_json({"error": f"engine error: {e}"}, status=500)

        # static UI (built) — fall back to index.html for client-side routing.
        return self._serve_static(path)

    def _serve_static(self, path):
        if path in ("/", ""):
            path = "/index.html"
        root = os.path.abspath(_UI_DIST)
        candidate = os.path.abspath(os.path.join(root, path.lstrip("/")))
        # Guard against path traversal: candidate must live under the UI root.
        if os.path.isfile(candidate) and os.path.commonpath([candidate, root]) == root:
            return self._send_file(candidate)
        index = os.path.join(root, "index.html")
        if os.path.isfile(index):
            return self._send_file(index)
        return self._send_json(
            {"error": "UI not built. Run `npm --prefix ui install && npm --prefix ui "
                      "run build`, or use the API endpoints under /api."}, status=404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()


def build_service(dataset_dir: Optional[str] = None, train_model: bool = True) -> EngineService:
    global _SERVICE
    _SERVICE = EngineService(dataset_dir=dataset_dir, train_model=train_model)
    return _SERVICE


def main() -> None:
    ap = argparse.ArgumentParser(description="Wharton ML engine API server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dataset", default=None, help="dataset dir (default: auto)")
    ap.add_argument("--no-model", action="store_true", help="skip ML training on boot")
    args = ap.parse_args()

    print("Loading engine service (data + model)…", flush=True)
    svc = build_service(args.dataset, train_model=not args.no_model)
    print(f"  dataset={os.path.basename(svc.dataset_dir)}  "
          f"names={len(svc.bundle.tickers)}  model={'yes' if svc.alpha_model else 'no'}",
          flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
