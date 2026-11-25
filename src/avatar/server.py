"""
Server local pentru pagina cu avatarul (doar biblioteca standard; merge și pe Raspberry Pi).

    python -m src.avatar.server                 # deschide http://127.0.0.1:8765 în browser
    python -m src.avatar.server --host 0.0.0.0  # pagina se poate deschide și de pe o tabletă din rețea
    python -m src.avatar.server --no-browser

Servește: pagina (src/avatar/web/), mișcările semnelor (/motion/<cuvânt>.json, din
data/avatar_motion/) și personajul (/avatar.vrm, din models/avatar/).
Pagina acceptă ?word=hurt (semnul de redat), ?sentence=you,hurt,what (semnele unei propoziții, legate
între ele; propozițiile de test sunt în web/sentences.js), ?t=1.2 (oprit la secunda 1.2, pentru capturi),
?points (scheletul din date lângă avatar) și ?check (în consolă: eroarea de direcție pe fiecare os).
Pornit din prototip (src/app/prototype.py), servește și /api/state (starea conversației, JSON),
/api/camera (imaginea camerei, MJPEG) și POST /api/command/<nume>; pagina le folosește cu ?live.
"""
import argparse
import functools
import http.server
import json
import shutil
import webbrowser
from pathlib import Path

from src.config import AVATAR_HOST, AVATAR_MODEL_FILE, AVATAR_MOTION_DIR, AVATAR_PORT

WEB_DIR = Path(__file__).parent / 'web'
# Edge / Chrome: capturile headless (evaluate_avatar.py) și fereastra prototipului (--app)
BROWSERS = ['msedge', 'chrome', 'chromium-browser', 'chromium', 'google-chrome',   # pe Pi: chromium(-browser)
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Google\Chrome\Application\chrome.exe']


def find_browser():
    for name in BROWSERS:
        path = shutil.which(name) or (name if Path(name).exists() else None)
        if path:
            return path
    raise SystemExit('Nu am găsit Edge sau Chrome (necesar pentru a reda avatarul).')


class AvatarHandler(http.server.SimpleHTTPRequestHandler):
    motion_dir = AVATAR_MOTION_DIR   # evaluate_avatar.py poate servi alt set de mișcări
    # Prototipul (src/app/prototype.py) leagă aici starea lui: snapshot() -> dict pentru
    # /api/state, frames() -> generator de imagini JPEG pentru /api/camera, command(nume).
    # Fără el (serverul pornit singur), rutele /api/ răspund 404.
    live = None

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if self.live is not None and path == '/api/state':
            self._send_json(self.live.snapshot())
        elif self.live is not None and path == '/api/camera':
            self._stream_camera()
        else:
            super().do_GET()

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if self.live is not None and path.startswith('/api/command/'):
            self.live.command(path.rsplit('/', 1)[-1])
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def _send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_camera(self):
        """Imaginea camerei ca MJPEG: browserul o afișează direct într-un <img>."""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=cadru')
        self.end_headers()
        try:
            for jpeg in self.live.frames():
                self.wfile.write(b'--cadru\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                 + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass   # pagina s-a închis sau s-a reîncărcat

    def translate_path(self, path):
        clean = path.split('?', 1)[0].split('#', 1)[0]
        if clean == '/avatar.vrm':
            return str(AVATAR_MODEL_FILE.resolve())
        if clean.startswith('/motion/'):
            name = Path(clean[len('/motion/'):]).name          # fără ../ : doar fișiere din folder
            return str((self.motion_dir / name).resolve())
        return super().translate_path(path)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')   # la dezvoltare: mereu ultima versiune
        super().end_headers()

    def log_message(self, fmt, *args):
        pass   # fără o linie în consolă pentru fiecare fișier


def make_server(host=AVATAR_HOST, port=AVATAR_PORT, handler_class=AvatarHandler):
    """Serverul paginii (încă nepornit: serve_forever). Întoarce (server, adresa paginii)."""
    handler_class.extensions_map.update({'.js': 'text/javascript', '.vrm': 'model/gltf-binary'})
    handler = functools.partial(handler_class, directory=str(WEB_DIR))
    server = http.server.ThreadingHTTPServer((host, port), handler)
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/"
    return server, url


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default=AVATAR_HOST)
    parser.add_argument('--port', type=int, default=AVATAR_PORT)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    for path, hint in ((AVATAR_MODEL_FILE, 'personajul VRM'),
                       (AVATAR_MOTION_DIR / 'index.json', 'rulează python -m src.avatar.export_motion')):
        if not path.exists():
            print(f"ATENȚIE: lipsește {path} ({hint})")

    server, url = make_server(args.host, args.port)
    print(f"Avatar: {url}  (Ctrl+C oprește serverul)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
