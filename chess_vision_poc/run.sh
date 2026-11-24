#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
if [ ! -f models/templates.npz ]; then
  .venv/bin/python -m chess_vision.train
fi
echo 'Open http://127.0.0.1:8765 in your browser'
exec .venv/bin/python -m chess_vision.app --open "$@"
