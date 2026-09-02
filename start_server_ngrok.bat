@echo off
echo Starting AI Duplicate Detection Service (Beta1)...
start "AI Duplicate Detection Backend" cmd /k "cd /d E:\Users\ai-duplicate-detection\Alpha && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
start "AI Duplicate Detection" cmd /k "ngrok http 8000"

echo Service is launching...
echo Swagger UI Docs: https://blast-zombie-riverboat.ngrok-free.dev/docs
