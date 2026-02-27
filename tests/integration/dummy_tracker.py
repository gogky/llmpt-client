from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = {
            "magnet_link": "magnet:?xt=urn:btih:3fa5dc5617bd5b7ccff37fd7e2ec80dcf25dc8eb&dn=gpt2_main&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337%2fannounce"
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))

if __name__ == "__main__":
    server = HTTPServer(("localhost", 8000), SimpleHandler)
    server.serve_forever()
