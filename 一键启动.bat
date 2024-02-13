:start
echo "Start VENV"
call Miniconda3\Scripts\activate.bat
goto :run

:run
echo "Start API + WebUI"
python startup.py -a

pause
