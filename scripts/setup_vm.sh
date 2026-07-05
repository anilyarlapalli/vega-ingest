#!/usr/bin/env bash
# One-shot vega bootstrap for a bare Debian/Ubuntu VM (no Docker).
#
#   scripts/setup_vm.sh            # CPU (Tesseract) install
#   scripts/setup_vm.sh --gpu      # + Surya/EasyOCR/torch (needs NVIDIA driver)
#   scripts/setup_vm.sh --gpu --warm   # also pre-download neural models now
#
# Idempotent: safe to re-run. Ends with `vega info` so the install
# self-verifies. GPU note: the only host prerequisite is the NVIDIA driver
# (`nvidia-smi` must work); torch's pip wheel bundles its own CUDA runtime.
set -euo pipefail

GPU=0 WARM=0
for arg in "$@"; do
    case "$arg" in
        --gpu)  GPU=1 ;;
        --warm) WARM=1 ;;
        *) echo "usage: $0 [--gpu] [--warm]" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."

echo "==> system packages (tesseract + Indic packs)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-venv python3-pip \
    tesseract-ocr tesseract-ocr-osd \
    tesseract-ocr-tel tesseract-ocr-hin tesseract-ocr-mar tesseract-ocr-tam \
    tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-ben tesseract-ocr-asm \
    tesseract-ocr-guj tesseract-ocr-pan tesseract-ocr-ori

echo "==> python venv + vega"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
if [ "$GPU" = 1 ]; then
    .venv/bin/pip install --quiet -e '.[gpu]'
else
    .venv/bin/pip install --quiet -e .
fi

if [ "$WARM" = 1 ]; then
    echo "==> warming neural models (first-use download, a few hundred MB)"
    .venv/bin/python - <<'EOF'
import io
from PIL import Image, ImageDraw

img = Image.new("RGB", (400, 80), "white")
ImageDraw.Draw(img).text((10, 20), "vega warm-up 1234", fill="black")
buf = io.BytesIO()
img.save(buf, "PNG")

from vega.ocr import select_backend
backend = select_backend("auto")
text = backend.image_to_text(buf.getvalue(), "eng")
print(f"warm-up OCR via {backend.name}: {text[:40]!r}")
EOF
fi

echo "==> verify"
.venv/bin/vega info
echo
echo "vega ready: .venv/bin/vega ingest <path> --lang <langs> --out chunks.jsonl"
