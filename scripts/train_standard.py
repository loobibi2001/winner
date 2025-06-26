# =====================================================================
# 專案: 機器學習策略專案
# 階段: 標準模型訓練腳本
#
# 功能: 使用固定參數快速訓練XGBoost模型
# =====================================================================

import os
import sys

# 添加主目錄到路徑
BASE_PATH = r"D:\飆股篩選\winner"
sys.path.append(BASE_PATH)

# 執行標準訓練腳本
if __name__ == "__main__":
    script_path = os.path.join(BASE_PATH, "V40.1_XGBoost模型訓練腳本_修正版.py")
    if os.path.exists(script_path):
        print("開始執行標準模型訓練...")
        exec(open(script_path, 'r', encoding='utf-8').read())
    else:
        print(f"錯誤：找不到訓練腳本 {script_path}") 