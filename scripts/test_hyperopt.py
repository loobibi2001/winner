# =====================================================================
# 專案: 機器學習策略專案
# 階段: 超參數調優測試腳本 (快速版本)
#
# 功能: 使用較少試驗次數快速測試超參數調優功能
# =====================================================================

import os
import sys

# 添加主目錄到路徑
BASE_PATH = r"D:\飆股篩選\winner"
SCRIPTS_DIR = os.path.join(BASE_PATH, "scripts")
sys.path.append(BASE_PATH)

if __name__ == "__main__":
    print("開始執行超參數調優測試...")
    
    # 讀取auto_hyperopt.py內容並修改試驗次數
    auto_hyperopt_path = os.path.join(SCRIPTS_DIR, "auto_hyperopt.py")
    
    if os.path.exists(auto_hyperopt_path):
        with open(auto_hyperopt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修改試驗次數
        content = content.replace('N_TRIALS = 50', 'N_TRIALS = 5')
        
        # 執行修改後的代碼
        exec(content)
    else:
        print(f"錯誤：找不到超參數調優腳本 {auto_hyperopt_path}") 