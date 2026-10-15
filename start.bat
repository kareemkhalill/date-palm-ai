@echo off
rem Starts Date Palm AI and opens it in the browser. Close this window to stop the server.
cd /d "%~dp0"

rem Already running? Just open the page.
curl -s -o nul http://127.0.0.1:8000/health && (start "" http://127.0.0.1:8000/ & exit /b)

rem Open the browser once the model has loaded (first start can take about a minute).
start "" /b powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){ try { if((Invoke-RestMethod http://127.0.0.1:8000/health).model_ready){ Start-Process 'http://127.0.0.1:8000/'; break } } catch {}; Start-Sleep 2 }"

echo Starting Date Palm AI on http://127.0.0.1:8000 ...
.venv\Scripts\python.exe app.py
pause
