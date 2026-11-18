"""
Server local pentru pagina cu avatarul (doar biblioteca standard; merge și pe Raspberry Pi).

    python -m src.avatar.server                 # deschide http://127.0.0.1:8765 în browser
    python -m src.avatar.server --host 0.0.0.0  # pagina se poate deschide și de pe o tabletă din rețea
    python -m src.avatar.server --no-browser

Servește: pagina (src/avatar/web/), mișcările semnelor (/motion/<cuvânt>.json, din
data/avatar_motion/) și personajul (/avatar.vrm, din models/avatar/).
Pagina acceptă ?word=hurt (semnul de redat), ?t=1.2 (oprit la secunda 1.2, pentru capturi),
?points (scheletul din date lângă avatar) și ?check (în consolă: eroarea de direcție pe fiecare os).
"""
import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

from src.config import AVATAR_HOST, AVATAR_MODEL_FILE, AVATAR_MOTION_DIR, AVATAR_PORT

WEB_DIR = Path(__file__).parent / 'web'


class AvatarHandler(http.server.SimpleHTTPRequestHandler):
    motion_dir = AVATAR_MOTION_DIR   # evaluate_avatar.py poate servi alt set de mișcări

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

    AvatarHandler.extensions_map.update({'.js': 'text/javascript', '.vrm': 'model/gltf-binary'})
    handler = functools.partial(AvatarHandler, directory=str(WEB_DIR))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    print(f"Avatar: {url}  (Ctrl+C oprește serverul)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
