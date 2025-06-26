@echo off
chcp 65001 > nul
cd /d "D:\飆股篩選\winner"
title 快速啟動交易儀表板

echo ===============================================================================
echo                      🌐 啟動AI交易儀表板 🌐
echo ===============================================================================
echo.
echo 📱 正在啟動交易儀表板，請稍候...
echo 💡 啟動後請在瀏覽器中開啟 http://localhost:8501
echo.

start /min cmd /c "python -m streamlit run AI_Web_Dashboard.py --server.port 8501"
timeout /t 3 > nul
start http://localhost:8501

echo ✅ 儀表板已啟動！
echo 🔗 瀏覽器應該會自動開啟，如果沒有請手動開啟：
echo    http://localhost:8501
echo.
echo 💡 提示：關閉此視窗不會停止儀表板服務
echo 🛑 若要停止服務，請按 Ctrl+C 或關閉對應的命令視窗
echo.
pause 