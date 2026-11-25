#!/usr/bin/env bash
# Verifică pe Pi fiecare componentă (după setup.sh); raportul ajunge în data/system_check/ și,
# dacă stick-ul e conectat, se copiază și pe el (ca să-l aduci înapoi pe laptop). Ex.:
#     ./check.sh --camera 1 --mic USB
cd "$(dirname "$(readlink -f "$0")")/../.."
export PYTHONIOENCODING=utf-8
[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || export DISPLAY=:0
venv/bin/python -m src.app.system_check "$@"
# stick-ul: montat automat de desktop (/media/...) sau manual, fără ecran (/mnt/stick, vezi CITESTE.md)
for stick in /media/"$USER"/*/licenta_pi /mnt/*/licenta_pi /mnt/licenta_pi; do
    [ -d "$stick" ] && mkdir -p "$stick/rapoarte" && cp data/system_check/*.txt "$stick/rapoarte/" 2>/dev/null \
        && echo "Rapoartele au fost copiate și pe stick: $stick/rapoarte/"
done
[ -t 0 ] && read -rp "Enter pentru a închide..."
exit 0
