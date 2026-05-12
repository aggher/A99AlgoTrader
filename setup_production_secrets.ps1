# setup_production_secrets.ps1 — Helper for Fly.io Deployment (Windows)

Write-Host "--- Setting Institutional Production Secrets ---" -ForegroundColor Cyan

$twelveData = Read-Host "Enter TwelveData API Key" -AsSecureString
$telegramToken = Read-Host "Enter Telegram Bot Token" -AsSecureString
$chatId = Read-Host "Enter Telegram Chat ID"

# Convert SecureStrings to plain text for the fly command
$twelveDataPlain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($twelveData))
$telegramTokenPlain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($telegramToken))

Write-Host "Pushing secrets to Fly.io..." -ForegroundColor Yellow
fly secrets set `
  TWELVE_DATA_API_KEY="$twelveDataPlain" `
  TELEGRAM_BOT_TOKEN="$telegramTokenPlain" `
  TELEGRAM_CHAT_ID="$chatId"

Write-Host "✅ Secrets successfully synchronized with cloud production." -ForegroundColor Green
