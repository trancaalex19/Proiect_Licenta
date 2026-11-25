#!/usr/bin/env bash
# Pornește prototipul pe Pi (după setup.sh). Argumentele trec mai departe, ex.:
#     ./run.sh --camera 1 --mode offline_only --mic USB
cd "$(dirname "$(readlink -f "$0")")/../.."
export PYTHONIOENCODING=utf-8
# pornit prin SSH: fereastra apare totuși pe ecranul Pi-ului
[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || export DISPLAY=:0
venv/bin/python -m src.app.prototype "$@"
status=$?
# din scurtătura de pe desktop: terminalul rămâne deschis, ca mesajele să se poată citi
[ -t 0 ] && read -rp "Prototipul s-a oprit (cod $status). Enter pentru a închide..."
exit $status
