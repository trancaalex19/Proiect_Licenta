# Prototipul pe Raspberry Pi

Pe stick, în folderul `licenta_pi/`, este tot ce trebuie: codul, modelele, un Python 3.11
portabil și toate pachetele pentru ARM64. **Pe Pi nu este nevoie de internet** pentru instalare.
Internetul e folosit doar de Groq (transcrierea online); fără el merge modelul Whisper local.

## Ce trebuie să ai

- Raspberry Pi 4 sau 5 cu **Raspberry Pi OS pe 64 de biți**, varianta cu desktop (Bookworm sau Trixie).
  Recomandat 8 GB RAM (merge și cu 4 GB) și un radiator sau ventilator.
- O **cameră USB**. O cameră pe panglică (Camera Module) nu merge direct cu OpenCV.
- Un microfon și un difuzor. O cameră web cu microfon încorporat e suficientă pentru microfon.
- Tastatură și mouse: Space înseamnă „vorbește medicul”.

## 0. Fără monitor, cameră sau microfon (Pi 4 doar cu sursa)

Obligatoriu: **card microSD** (32 GB, clasa A1/A2) și sursa. Restul se înlocuiește.

**Sistemul pe card** (de pe laptop): [Raspberry Pi Imager](https://www.raspberrypi.com/software/) →
Raspberry Pi 4 → **Raspberry Pi OS (64-bit)** → cardul. La „Customisation” setezi un nume de
utilizator și o parolă, **Wi-Fi-ul** (aceeași rețea ca laptopul și telefonul) și **SSH activat**.
Cardul în Pi, alimentarea, apoi așteaptă ~2 minute.

**Conectarea de pe laptop** (PowerShell): `ssh <utilizator>@raspberrypi.local`
(sau numele pus în Imager). Fără desktop, stick-ul nu se montează singur:

```bash
lsblk                                   # stick-ul apare ca sda (sda1 = partiția)
sudo mkdir -p /mnt/stick && sudo mount -o uid=$USER,gid=$USER /dev/sda1 /mnt/stick
bash /mnt/stick/licenta_pi/setup.sh
```

**Ecranul = browserul laptopului**: `~/licenta/deploy/pi/run.sh --host 0.0.0.0 --no-browser`,
apoi pe laptop `http://raspberrypi.local:8765/?live` (Space și butonul merg din pagină).

**Camera = telefonul prin Wi-Fi**: aplicația **IP Webcam** (Android; „Start server”, rezoluție
640x480) sau **DroidCam** (Android/iPhone). Adresa o arată aplicația:

```bash
~/licenta/deploy/pi/check.sh --camera http://192.168.1.23:8080/video     # IP Webcam
~/licenta/deploy/pi/run.sh --camera http://192.168.1.23:8080/video --host 0.0.0.0 --no-browser
# DroidCam: http://<ip>:4747/video
```

Telefonul ca cameră merge la fel și pe laptop: `python -m src.app.prototype --camera http://...`.

**Microfon + difuzor = căști Bluetooth** (ex. JBL Wave Beam 2, au microfon). Mai sigur: o cameră
web USB cu microfon sau căști USB.

```bash
bluetoothctl         # apoi: scan on -> (căștile în modul de împerechere) -> pair <MAC> -> trust <MAC> -> connect <MAC> -> quit
~/licenta/deploy/pi/check.sh --mic JBL    # microfonul după o parte din nume
```

## 1. Instalare (o singură dată, ~5-10 minute)

Conectează stick-ul, deschide un terminal și rulează:

```bash
bash /media/$USER/*/licenta_pi/setup.sh
```

Proiectul ajunge în `~/licenta`. Dacă lipsește PortAudio, scriptul cere parola de `sudo`.
Rulat din nou, actualizează codul și păstrează ce s-a înregistrat pe Pi.
Jurnalul instalării: `licenta_pi/setup_log.txt` (pe stick).

## 2. Verificare (înainte de prototip)

```bash
~/licenta/deploy/pi/check.sh
```

Sau dublu-clic pe **„Verificare sistem”** de pe desktop. Verificarea testează pe rând:
RAM și temperatura, modelele, recunoașterea semnelor, camera (cu MediaPipe, FPS),
microfonul (vorbește 3 s), vocea Piper (o auzi în difuzor), Whisper local, Groq și Chromium.
Raportul se salvează în `~/licenta/data/system_check/` și se copiază pe stick, în
`licenta_pi/rapoarte/`, ca să-l aduci înapoi pe laptop.

| Problemă în raport | Ce faci |
|---|---|
| camera nu se deschide | `check.sh --camera 1` (sau 2); camerele: `ls /dev/video*` |
| microfon greșit / nivel foarte mic | numele apar în listă: `check.sh --mic USB` (o parte din nume) |
| nu se aude vocea | clic dreapta pe iconița de volum (sus, dreapta) și alege ieșirea audio |
| FPS MediaPipe sub 8 | în `~/licenta/src/config.py`: `HOLISTIC_MODEL_COMPLEXITY = 0` |
| Whisper local lent (RTF peste 1.5) | normal pe Pi 4; cu internet, Groq face transcrierea |
| `vcgencmd ... throttled` diferit de 0x0 | sursă de alimentare prea slabă sau Pi-ul se încălzește prea tare |

## 3. Prototipul

```bash
~/licenta/deploy/pi/run.sh
```

Sau dublu-clic pe **„Prototip licență”**. Opțiunile merg la fel ca pe laptop:
`run.sh --camera 1 --mic USB --mode offline_only`. Închiderea ferestrei oprește tot.
Conversația se salvează în `~/licenta/data/prototype_log.csv`.

Dacă fereastra merge greu pe Pi (avatarul 3D), pagina se poate deschide de pe o tabletă sau
de pe laptop din aceeași rețea Wi-Fi:

```bash
~/licenta/deploy/pi/run.sh --host 0.0.0.0 --no-browser
hostname -I        # adresa Pi-ului; pe tabletă: http://<adresa>:8765/?live
```

## Pentru laptop (pregătirea stick-ului)

```powershell
python deploy/pi/make_stick.py --download    # o dată: pachetele Pi în dist/pi/ (~1 GB)
python deploy/pi/make_stick.py E:\           # tot pe stick (cu modificările de acum din cod)
```

Pe stick ajunge și `.env`, cu cheia Groq. Nu da stick-ul altcuiva fără să ștergi cheia.
