@echo off
echo ================================
echo Weeing-weeing 환경 설정 시작
echo ================================
echo.

REM Python 설치 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python이 설치되어 있지 않습니다.
    echo Python 3.8 이상을 설치해주세요: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Python 버전 확인...
python --version
echo.

REM 가상환경 생성
echo [2/3] 가상환경 생성중...
if not exist "venv" (
    python -m venv venv
    echo 가상환경이 생성되었습니다.
) else (
    echo 가상환경이 이미 존재합니다.
)
echo.

REM 가상환경 활성화 및 패키지 설치
echo [3/3] 패키지 설치중...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

echo ================================
echo 설정 완료!
echo ================================
echo.
echo 실행 방법:
echo   1. venv\Scripts\activate.bat  (가상환경 활성화)
echo   2. streamlit run app.py       (앱 실행)
echo.
echo 또는 run.bat을 실행하세요.
echo.
pause

