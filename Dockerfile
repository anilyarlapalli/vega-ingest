# vega CPU image — the Tesseract parse+chunk path, no GPU.
#
#   docker build -t vega .
#   docker run --rm -v ./corpus:/data -v ./out:/out -v vega-cache:/cache \
#       vega ingest /data --lang ta --out /out/chunks.jsonl
#
# See README "Deployment" for the volume layout and env knobs.
FROM python:3.12-slim

# Tesseract engine + OSD + every Indic pack vega supports. System packages —
# the same apt set the README's Install section documents.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-osd \
        tesseract-ocr-tel tesseract-ocr-hin tesseract-ocr-mar tesseract-ocr-tam \
        tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-ben tesseract-ocr-asm \
        tesseract-ocr-guj tesseract-ocr-pan tesseract-ocr-ori \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md constraints.txt ./
COPY vega/ vega/
# -c constraints.txt: known-good lockfile (A1) — pins resolution without
# adding packages, so a rebuild can't silently drift dependency versions.
RUN pip install --no-cache-dir . -c constraints.txt

# OCR cache lives on the /cache volume so it survives container restarts and
# is shared by parallel containers (writes are atomic).
ENV VEGA_OCR_CACHE_DIR=/cache/ocr
RUN useradd -m vega && mkdir -p /cache /data /out \
    && chown -R vega:vega /cache /data /out
USER vega

ENTRYPOINT ["vega"]
CMD ["--help"]
