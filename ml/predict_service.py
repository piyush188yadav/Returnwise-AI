"""
ReturnWise AI - ML Microservice HTTP Server
============================================
Lightweight, resilient HTTP server serving the scikit-learn return risk model.
Uses Python standard http.server with JSON request/response handling.
Default port: 5001
"""

import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add current project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.predict import predict_return_risk, load_model


class MLServiceHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        if self.path == "/health":
            pipeline = load_model()
            has_model = pipeline is not None
            resp = {
                "status": "healthy",
                "service": "ReturnWise AI Python ML Prediction Engine",
                "model_loaded": has_model,
                "port": 5001
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        if self.path == "/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length == 0:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "Empty request body"}).encode("utf-8"))
                    return

                post_data = self.rfile.read(content_length)
                order_payload = json.loads(post_data.decode("utf-8"))

                prediction = predict_return_risk(order_payload)
                self._set_headers(200)
                self.wfile.write(json.dumps(prediction).encode("utf-8"))
            except json.JSONDecodeError:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Malformed JSON payload"}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def log_message(self, format, *args):
        # Concise logging
        sys.stdout.write(f"[ML-SERVICE] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()


def run(port=5001):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, MLServiceHandler)
    print(f"[ML-SERVICE] ReturnWise AI ML Service listening on http://127.0.0.1:{port}")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ML-SERVICE] Shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", 5001))
    run(port)
