@echo off
REM Start offline LLM & RAG backend and frontend

REM 1. Start backend (prefer server.exe, fallback to python server.py)
if exist server.exe (
    start "Offline LLM-RAG Backend" server.exe
) else (
    start "Offline LLM-RAG Backend" python server.py
)

REM 2. Open frontend (web_folder/index.html)
start "Offline LLM-RAG Frontend" "web_folder\index.html"

REM 3. Optional: auto close batch window
exit
