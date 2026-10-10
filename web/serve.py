"""Static file server for the demo (serves the project root). Usage: python web/serve.py [port]"""
import http.server
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                      ".wasm": "application/wasm", ".onnx": "application/octet-stream"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT))).serve_forever()
