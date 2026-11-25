"""
Pregătește stick-ul pentru Raspberry Pi (rulat pe laptop, din rădăcina proiectului):

    python deploy/pi/make_stick.py --download     # o dată (internet): pachetele Pi în dist/pi/
    python deploy/pi/make_stick.py E:\            # copiază totul pe stick, în E:\licenta_pi\

Pe stick ajung: project.tar (codul, datele din git și cele noi, modelele care nu sunt în git:
Whisper local, vocea Piper, avatarul; plus .env cu cheia Groq), Python 3.11 portabil pentru ARM64,
toate pachetele (wheel-uri aarch64, deci pe Pi nu trebuie internet), PortAudio (.deb) și
setup.sh + CITESTE.md. Pe Pi: bash /media/$USER/<STICK>/licenta_pi/setup.sh (vezi CITESTE.md).
"""
import argparse
import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DIST = ROOT / 'dist' / 'pi'
LOCK = HERE / 'requirements-pi.lock.txt'
# Python portabil (python-build-standalone, aceeași sursă ca uv); 3.11 = versiunea din requirements.txt
PYTHON_URL = ('https://releases.astral.sh/github/python-build-standalone/releases/download/20261110/'
              'cpython-3.11.18%2B20261110-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz')
PYTHON_FILE = 'python-3.11-aarch64.tar.gz'
# PortAudio (microfon / difuzor) cu toate dependențele lui (libasound2, libjack...), pentru Raspberry Pi
# OS Bookworm (Debian 12) și Trixie (Debian 13), dacă Pi-ul n-are internet. Se descarcă prin Docker
# (Debian arm64 minimal), deci e un set complet; setup.sh instalează pe Pi doar ce lipsește.
DEB_RELEASES = ['bookworm', 'trixie']
PLATFORMS = ['manylinux2014_aarch64'] + [f'manylinux_2_{v}_aarch64' for v in (17, 24, 26, 27, 28, 31, 34, 35)] \
    + ['linux_aarch64', 'any']
# Ce nu este în git, dar trebuie pe Pi
EXTRA = ['models/whisper-ro-ct2', 'models/piper', 'models/avatar', '.env']
# În git / lângă proiect, dar inutile pe Pi
SKIP_PREFIXES = ('dist/', 'deploy/pi/make_stick.py')
SKIP_ROOT_SUFFIXES = ('.png', '.jpg')        # capturile de ecran din rădăcină


def download():
    """Wheel-urile aarch64 / Python 3.11 din fișierul lock, unul câte unul (se reia de unde a rămas)."""
    wheels = DIST / 'wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    flags = ['--only-binary=:all:', '--python-version', '3.11', '--implementation', 'cp',
             '--abi', 'cp311', '--abi', 'abi3', '--abi', 'none', '--no-deps', '-d', str(wheels),
             '--timeout', '60', '--retries', '5', '-q']
    for platform in PLATFORMS:
        flags += ['--platform', platform]
    have = {p.name.split('-')[0].lower().replace('_', '-') for p in wheels.glob('*.whl')}
    failed = []
    for line in LOCK.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        if line.split('==')[0].lower().replace('_', '-') in have:
            continue
        print(f'  {line}')
        if subprocess.run([sys.executable, '-m', 'pip', 'download', line, *flags]).returncode != 0:
            failed.append(line)
    if not (DIST / PYTHON_FILE).exists():
        print(f'  {PYTHON_FILE}')
        with urllib.request.urlopen(PYTHON_URL, timeout=60) as response, open(DIST / PYTHON_FILE, 'wb') as f:
            shutil.copyfileobj(response, f)
    for release in DEB_RELEASES:
        out = DIST / 'debs' / release
        if any(out.glob('libportaudio2_*.deb')):
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f'  PortAudio + dependențe pentru {release} (Docker, arm64)')
        command = ('apt-get update -qq && apt-get install -y -qq --download-only --no-install-recommends '
                   'libportaudio2 >/dev/null && cp /var/cache/apt/archives/*.deb /out/')
        if not shutil.which('docker') or subprocess.run(
                ['docker', 'run', '--rm', '--platform', 'linux/arm64', '-v', f'{out}:/out', f'debian:{release}',
                 'bash', '-c', command]).returncode != 0:
            failed.append(f'debs/{release} (pornește Docker Desktop)')
    print(f"Pachete: {len(list(wheels.glob('*.whl')))} wheel-uri în {wheels}"
          + (f"; NU s-au descărcat: {', '.join(failed)}" if failed else ''))
    return not failed


def project_files():
    """Fișierele proiectului: cele din git + cele noi neignorate + modelele mari (EXTRA)."""
    out = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=ROOT,
                         capture_output=True, text=True, encoding='utf-8', check=True).stdout
    files = [f for f in out.split('\0') if f and (ROOT / f).is_file()]
    files = [f for f in files if not f.startswith(SKIP_PREFIXES)
             and not ('/' not in f and f.lower().endswith(SKIP_ROOT_SUFFIXES))]
    for extra in EXTRA:
        path = ROOT / extra
        if path.is_dir():
            files += [p.relative_to(ROOT).as_posix() for p in path.rglob('*') if p.is_file()]
        elif path.is_file():
            files.append(extra)
        else:
            print(f'ATENȚIE: lipsește {extra} (nu ajunge pe Pi)')
    return sorted(set(files))


def write_tar(files, path):
    """project.tar: scripturile .sh cu LF și drept de execuție (Windows le poate avea cu CRLF)."""
    with tarfile.open(path, 'w') as tar:
        for name in files:
            source = ROOT / name
            info = tar.gettarinfo(str(source), arcname=name)
            if name.endswith('.sh'):
                data = source.read_bytes().replace(b'\r\n', b'\n')
                info.size, info.mode = len(data), 0o755
                tar.addfile(info, io.BytesIO(data))
            else:
                info.mode = 0o644
                with open(source, 'rb') as f:
                    tar.addfile(info, f)


def copy_tree(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob('*'):
        if path.is_file():
            dest = target / path.relative_to(source)
            if not dest.exists() or dest.stat().st_size != path.stat().st_size:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)


def main():
    parser = argparse.ArgumentParser(description='Pregătește stick-ul pentru Raspberry Pi')
    parser.add_argument('target', nargs='?', help='rădăcina stick-ului (ex. E:\\) sau orice folder')
    parser.add_argument('--download', action='store_true', help='descarcă întâi pachetele Pi în dist/pi/')
    args = parser.parse_args()
    if args.download and not download():
        sys.exit('Unele pachete nu s-au descărcat: rulează din nou (se reia de unde a rămas).')
    if not args.target:
        return
    if not (DIST / PYTHON_FILE).exists() or not any((DIST / 'wheels').glob('*.whl')):
        sys.exit('Lipsesc pachetele Pi: rulează întâi python deploy/pi/make_stick.py --download')

    out = Path(args.target) / 'licenta_pi'
    out.mkdir(parents=True, exist_ok=True)
    files = project_files()
    print(f'1/3 project.tar: {len(files)} fișiere ...')
    write_tar(files, out / 'project.tar')
    print('2/3 pachetele (wheel-uri, Python 3.11, PortAudio) ...')
    copy_tree(DIST / 'wheels', out / 'wheels')
    copy_tree(DIST / 'debs', out / 'debs')
    shutil.copy2(DIST / PYTHON_FILE, out / PYTHON_FILE)
    print('3/3 setup.sh și CITESTE.md ...')
    (out / 'setup.sh').write_bytes((HERE / 'setup.sh').read_bytes().replace(b'\r\n', b'\n'))
    shutil.copy2(HERE / 'CITESTE.md', out / 'CITESTE.md')
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1024 ** 3
    print(f'Gata: {out} ({size:.2f} GB). Pe Pi: bash /media/$USER/<STICK>/licenta_pi/setup.sh')


if __name__ == '__main__':
    main()
