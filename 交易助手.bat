@echo off
chcp 65001 > nul
cd /d "D:\飆股篩選\winner"
title AI股票交易助手

:MAIN_MENU
cls
echo ===============================================================================
echo                      🤖 AI股票交易助手 v2.0 🤖
echo ===============================================================================
echo.
echo 請選擇您要執行的功能：
echo.
echo [1] 🚀 一鍵完整流程 (更新資料 + 訓練模型 + 回測 + 啟動儀表板)
echo [2] 📊 只更新股票資料
echo [3] 🤖 只訓練AI模型
echo [4] 📈 只執行回測分析
echo [5] 🌐 只啟動交易儀表板
echo [6] 📋 查看最新回測報告
echo [7] 🔧 系統資訊檢查
echo [0] ❌ 退出程式
echo.
echo ===============================================================================
set /p choice="請輸入選項 (0-7): "

if "%choice%"=="1" goto FULL_PROCESS
if "%choice%"=="2" goto UPDATE_DATA
if "%choice%"=="3" goto TRAIN_MODEL
if "%choice%"=="4" goto BACKTEST
if "%choice%"=="5" goto DASHBOARD
if "%choice%"=="6" goto VIEW_REPORT
if "%choice%"=="7" goto SYSTEM_INFO
if "%choice%"=="0" goto EXIT
goto INVALID_CHOICE

:FULL_PROCESS
echo.
echo ⏳ 正在執行完整流程，這可能需要30-60分鐘...
echo.
python scripts/train_and_backtest.py
pause
goto MAIN_MENU

:UPDATE_DATA
echo.
echo 📊 正在更新股票資料...
echo.
python scripts/update_all_data.py
pause
goto MAIN_MENU

:TRAIN_MODEL
echo.
echo 🤖 正在訓練AI模型...
echo.
python "V40.1_XGBoost模型訓練腳本_修正版.py"
pause
goto MAIN_MENU

:BACKTEST
echo.
echo 📈 正在執行回測分析...
echo.
python "V40.1_XGBoost回測最終版.py"
pause
goto MAIN_MENU

:DASHBOARD
echo.
echo 🌐 正在啟動交易儀表板...
echo 📱 請稍後在瀏覽器中開啟 http://localhost:8501
echo.
start /min cmd /c "python -m streamlit run AI_Web_Dashboard.py --server.port 8501"
timeout /t 3 > nul
start http://localhost:8501
echo.
echo ✅ 儀表板已啟動！請在瀏覽器中查看
pause
goto MAIN_MENU

:VIEW_REPORT
echo.
echo 📋 查看最新回測報告...
echo.
if exist "runs\" (
    for /f "delims=" %%i in ('dir "runs" /b /o-d /ad 2^>nul') do (
        if exist "runs\%%i\kpi_report.txt" (
            echo ======= 最新回測報告 (%%i) =======
            type "runs\%%i\kpi_report.txt"
            goto REPORT_FOUND
        )
    )
    echo ❌ 找不到回測報告，請先執行回測分析
) else (
    echo ❌ 找不到 runs 目錄，請先執行回測分析
)
:REPORT_FOUND
pause
goto MAIN_MENU

:SYSTEM_INFO
echo.
echo 🔧 系統資訊檢查...
echo.
echo === Python 版本 ===
python --version
echo.
echo === 必要套件檢查 ===
python -c "import pandas, numpy, xgboost, sklearn, streamlit; print('✅ 所有套件正常')" 2>nul || echo "❌ 部分套件缺失"
echo.
echo === 模型檔案檢查 ===
if exist "xgboost_long_short_model.joblib" (echo ✅ XGBoost模型檔案存在) else (echo ❌ XGBoost模型檔案不存在)
if exist "regime_model.joblib" (echo ✅ 市場模型檔案存在) else (echo ❌ 市場模型檔案不存在)
echo.
echo === 資料目錄檢查 ===
if exist "StockData_Parquet\" (echo ✅ 股票資料目錄存在) else (echo ❌ 股票資料目錄不存在)
if exist "stock_list.txt" (echo ✅ 股票清單檔案存在) else (echo ❌ 股票清單檔案不存在)
pause
goto MAIN_MENU

:INVALID_CHOICE
echo.
echo ❌ 無效的選項，請重新選擇！
timeout /t 2 > nul
goto MAIN_MENU

:EXIT
echo.
echo 👋 感謝使用AI股票交易助手！
timeout /t 2 > nul
exit 