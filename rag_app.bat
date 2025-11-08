@echo off
echo Starting Streamlit app using venv Python...
call "%~dp0venv\Scripts\activate"
python -m streamlit run rag_main.py
pause
