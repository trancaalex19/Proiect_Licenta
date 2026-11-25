#!/usr/bin/env bash
# Instalează prototipul pe Raspberry Pi (64 de biți) direct de pe stick, fără internet:
#     bash /media/$USER/<STICK>/licenta_pi/setup.sh            # în ~/licenta
#     bash .../setup.sh /alt/folder
# Rulat din nou: actualizează codul și pachetele, păstrează ce s-a înregistrat pe Pi (data/).
# Nu depinde de Python-ul sistemului: Raspberry Pi OS Trixie are Python 3.13, iar TensorFlow 2.15
# și MediaPipe 0.10.7 merg doar până la 3.11, așa că folosim un Python 3.11 portabil de pe stick.
set -euo pipefail

STICK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HOME/licenta}"
LOG="$STICK/setup_log.txt"
exec > >(tee "$LOG") 2>&1
echo "Instalare prototip: $(date '+%Y-%m-%d %H:%M'), $(hostname), destinația $DEST"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "EROARE: e nevoie de Raspberry Pi OS pe 64 de biți (aarch64); sistemul este $(uname -m)."
    exit 1
fi
. /etc/os-release
echo "Sistem: $PRETTY_NAME"

echo
echo "[1/5] Copiez proiectul în $DEST ..."
mkdir -p "$DEST"
tar -xf "$STICK/project.tar" -C "$DEST"

echo
echo "[2/5] Python 3.11 portabil ..."
PY="$DEST/.python/bin/python3.11"
if [ ! -x "$PY" ]; then
    mkdir -p "$DEST/.python"
    tar -xzf "$STICK/python-3.11-aarch64.tar.gz" -C "$DEST/.python" --strip-components=1
fi
"$PY" --version

echo
echo "[3/5] Mediul virtual și pachetele (de pe stick, fără internet) ..."
[ -x "$DEST/venv/bin/python" ] || "$PY" -m venv "$DEST/venv"
PIP="$DEST/venv/bin/python -m pip"
if ! $PIP install --no-index --find-links "$STICK/wheels" -r "$DEST/deploy/pi/requirements-pi.lock.txt"; then
    echo "Pachetele de pe stick nu au ajuns: încerc de pe internet ..."
    $PIP install -r "$DEST/deploy/pi/requirements-pi.lock.txt"
fi

echo
echo "[4/5] Biblioteci de sistem ..."
if /sbin/ldconfig -p | grep -q 'libportaudio.so.2'; then
    echo "PortAudio (microfon/difuzor): există"
else
    echo "PortAudio lipsește: îl instalez (cere parola de sudo) ..."
    if ! sudo apt-get install -y libportaudio2; then
        # Fără internet: de pe stick, PortAudio și dependențele lui, dar DOAR pachetele care lipsesc
        # (cele deja instalate pe Pi pot fi mai noi și nu le înlocuim)
        dir="$STICK/debs/${VERSION_CODENAME:-bookworm}"
        [ -d "$dir" ] || dir="$STICK/debs/bookworm"
        missing=()
        for deb in "$dir"/*.deb; do
            pkg=$(dpkg-deb -f "$deb" Package)
            if [ "$(dpkg-query -W -f='${db:Status-Status}' "$pkg" 2>/dev/null)" != "installed" ]; then
                missing+=("$deb")
            fi
        done
        echo "Fără internet: instalez de pe stick ${#missing[@]} pachete din $dir"
        sudo dpkg -i "${missing[@]}" || echo "ATENȚIE: PortAudio nu s-a instalat; microfonul și vocea nu vor merge."
    fi
fi
if command -v chromium-browser >/dev/null || command -v chromium >/dev/null; then
    echo "Chromium (fereastra prototipului): există"
else
    echo "ATENȚIE: Chromium lipsește (sudo apt install chromium). Fără el: run.sh --no-browser + altă tabletă/browser."
fi
/sbin/ldconfig -p | grep -q 'libGL.so.1' || echo "ATENȚIE: lipsește libGL (OpenCV): sudo apt install libgl1"

echo
echo "[5/5] Scurtături ..."
chmod +x "$DEST"/deploy/pi/*.sh
if [ -d "$HOME/Desktop" ]; then
    for name in run check; do
        title=$([ $name = run ] && echo "Prototip licență" || echo "Verificare sistem")
        cat > "$HOME/Desktop/licenta-$name.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$title
Exec=$DEST/deploy/pi/$name.sh
Path=$DEST
Terminal=true
Icon=utilities-terminal
EOF
        chmod +x "$HOME/Desktop/licenta-$name.desktop"
    done
    echo "Pe desktop: „Prototip licență” și „Verificare sistem”"
fi

echo
echo "Gata. Pași următori:"
echo "  $DEST/deploy/pi/check.sh     # verifică fiecare componentă (camera, microfon, voce, Whisper...)"
echo "  $DEST/deploy/pi/run.sh       # prototipul"
echo "Jurnalul instalării: $LOG"
