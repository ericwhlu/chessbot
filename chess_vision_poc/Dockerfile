FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 STOCKFISH_PATH=/usr/games/stockfish
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-hosting.txt ./
RUN pip install --no-cache-dir -r requirements-hosting.txt
COPY chess_vision/ chess_vision/
COPY static/ static/
COPY models/ models/
RUN python -m chess_vision.render
COPY gunicorn.conf.py ./
RUN useradd --create-home --uid 10001 chessmate
USER chessmate
EXPOSE 8765
CMD ["gunicorn", "--config", "gunicorn.conf.py", "chess_vision.app:create_app()"]
