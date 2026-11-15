"""Synthetic HTTP child for native lifecycle tests; never loads model weights."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
p = argparse.ArgumentParser()
p.add_argument('--alias', required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--unhealthy', action='store_true')
a = p.parse_args()
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def do_GET(self):
        body = json.dumps({'status': 'loading' if a.unhealthy else 'ok'} if self.path == '/health'
                          else {'data': [{'id': a.alias}]}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
HTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
