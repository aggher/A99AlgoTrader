#!/bin/bash
# setup_production_secrets.sh — Helper for Fly.io Deployment

echo "--- Setting Institutional Production Secrets ---"

# Prompt for secrets (do not echo for security)
read -sp "Enter TwelveData API Key: " TWELVE_DATA_API_KEY
echo
read -sp "Enter Telegram Bot Token: " TELEGRAM_BOT_TOKEN
echo
read -p "Enter Telegram Chat ID: " TELEGRAM_CHAT_ID

# Set secrets on Fly.io
echo "Pushing secrets to Fly.io..."
fly secrets set \
  TWELVE_DATA_API_KEY="$TWELVE_DATA_API_KEY" \
  TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID"

echo "✅ Secrets successfully synchronized with cloud production."
