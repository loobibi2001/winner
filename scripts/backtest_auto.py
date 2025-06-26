# =====================================================================
# 專案: 機器學習策略專案
# 階段: 自動回測腳本
#
# 功能: 自動執行回測並產生報告
# =====================================================================

import os
import sys

# 添加主目錄到路徑
BASE_PATH = r"D:\飆股篩選\winner"
sys.path.append(BASE_PATH)

# 執行回測腳本
if __name__ == "__main__":
    script_path = os.path.join(BASE_PATH, "V40.1_XGBoost回測最終版.py")
    if os.path.exists(script_path):
        print("開始執行自動回測...")
        exec(open(script_path, 'r', encoding='utf-8').read())
    else:
        print(f"錯誤：找不到回測腳本 {script_path}") 