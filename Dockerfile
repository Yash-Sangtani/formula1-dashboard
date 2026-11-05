FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    F1_MODE=sim PORT=8000 HOST=0.0.0.0

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# Recordings (raw live feed) and FastF1 cache live on a volume so they survive restarts.
RUN useradd -m app && mkdir -p /app/recordings && chown -R app /app
USER app
VOLUME ["/app/recordings"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz'%os.environ.get('PORT','8000'),timeout=4)"

CMD ["python", "-m", "backend"]
