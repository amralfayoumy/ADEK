#!/usr/bin/env bash
# run.sh – Start Streamlit and expose via Cloudflare Tunnel
# Usage:  bash run.sh

set -e

PORT=8501
APP="app.py"

echo "=========================================="
echo "  Student Risk Analytics Dashboard"
echo "=========================================="

# ── 1. Check Python deps ──────────────────────────────────────────────────────
echo ""
echo "▶ Installing/verifying Python dependencies …"
pip install -r requirements.txt --quiet

# ── 2. Check dataset ──────────────────────────────────────────────────────────
if [ ! -f "data.csv" ]; then
  echo ""
  echo "⚠️  data.csv not found!"
  echo ""
  echo "Download the dataset from one of these sources:"
  echo "  Kaggle competition : https://www.kaggle.com/competitions/playground-series-s4e6/data"
  echo "                       → download 'train.csv' and rename to 'data.csv'"
  echo "  UCI / Kaggle dataset: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention"
  echo "                       → download 'data.csv'"
  echo ""
  echo "Place data.csv in this folder, then re-run: bash run.sh"
  exit 1
fi

# ── 3. Launch Streamlit (background) ─────────────────────────────────────────
echo ""
echo "▶ Starting Streamlit on port $PORT …"
streamlit run "$APP" \
  --server.port $PORT \
  --server.headless true \
  --server.fileWatcherType none \
  &
STREAMLIT_PID=$!
echo "  Streamlit PID: $STREAMLIT_PID"

# Give Streamlit a few seconds to boot
sleep 4

# ── 4. Cloudflare Tunnel ──────────────────────────────────────────────────────
echo ""
echo "▶ Starting Cloudflare Tunnel …"
echo "  (if cloudflared is not installed, see install instructions below)"
echo ""

if command -v cloudflared &> /dev/null; then
  cloudflared tunnel --url http://localhost:$PORT
else
  echo "  cloudflared not found. Install it:"
  echo ""
  echo "  macOS:   brew install cloudflare/cloudflare/cloudflared"
  echo "  Linux:   curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb"
  echo "           sudo dpkg -i cloudflared.deb"
  echo "  Windows: winget install Cloudflare.cloudflared"
  echo ""
  echo "  Or use ngrok as an alternative:"
  echo "    ngrok http $PORT"
  echo ""
  echo "Dashboard is still running locally at: http://localhost:$PORT"
  echo "Press Ctrl+C to stop."
  wait $STREAMLIT_PID
fi
