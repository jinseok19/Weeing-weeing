@echo off
echo Weeing-weeing 앱 실행중...
echo.

REM 가상환경 확인
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] 가상환경이 없습니다.
    echo setup.bat을 먼저 실행해주세요.
    pause
    exit /b 1
)

REM 가상환경 활성화 및 앱 실행
call venv\Scripts\activate.bat
streamlit run app.py

