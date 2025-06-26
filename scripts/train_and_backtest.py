# =====================================================================
# 專案: 機器學習策略專案
# 階段: 一鍵自動化訓練與回測腳本 (含超參數調優)
#
# 功能:
#   - 一鍵更新資料
#   - 一鍵訓練模型 (可選擇是否進行超參數調優)
#   - 一鍵執行回測
#   - 自動產生報告與圖表
# =====================================================================

import os
import sys
import subprocess
import logging
import time
from tqdm import tqdm

# 設定路徑
BASE_PATH = r"D:\飆股篩選\winner"
SCRIPTS_DIR = os.path.join(BASE_PATH, "scripts")

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path, description):
    """執行腳本並即時顯示輸出"""
    logger.info(f"開始執行: {description}")
    
    # 構建完整路徑
    if not os.path.isabs(script_path):
        full_path = os.path.join(BASE_PATH, "scripts", script_path)
    else:
        full_path = script_path
    
    logger.info(f"腳本路徑: {full_path}")
    
    if not os.path.exists(full_path):
        logger.error(f"腳本不存在: {full_path}")
        return False
    
    try:
        # 直接執行，不捕獲輸出，讓輸出直接顯示到控制台
        result = subprocess.run(
            [sys.executable, full_path], 
            cwd=BASE_PATH,
            timeout=1800  # 30分鐘超時
        )
        
        if result.returncode == 0:
            logger.info(f"✓ {description} 執行成功")
            return True
        else:
            logger.error(f"✗ {description} 執行失敗，返回碼: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} 執行超時")
        return False
    except Exception as e:
        logger.error(f"✗ 執行 {description} 時發生錯誤: {e}")
        return False

def main():
    """主要執行函數，帶有整體進度條"""
    print("=" * 80)
    print("            📈 AI股票交易系統 - 完整訓練與回測流程")
    print("=" * 80)
    
    # 定義所有步驟
    steps = [
        ("update_all_data.py", "更新股票資料", 30),
        ("../V40.1_XGBoost模型訓練腳本_修正版.py", "XGBoost模型訓練", 25),
        ("../V40.1_XGBoost回測最終版.py", "執行回測分析", 25),
        ("../AI_Web_Dashboard.py", "啟動交易儀表板", 20)
    ]
    
    total_steps = len(steps)
    overall_start_time = time.time()
    
    # 整體進度條
    with tqdm(total=100, desc="整體進度", unit="%", 
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed} < {remaining}") as pbar:
        
        current_progress = 0
        
        for i, (script_path, description, weight) in enumerate(steps):
            # 更新進度條描述
            pbar.set_description(f"[{i+1}/{total_steps}] {description}")
            
            step_start_time = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"步驟 {i+1}/{total_steps}: {description}")
            logger.info(f"{'='*60}")
            
            # 執行步驟
            success = run_script(script_path, description)
            
            step_elapsed = time.time() - step_start_time
            logger.info(f"步驟耗時: {step_elapsed/60:.1f} 分鐘")
            
            if not success:
                pbar.set_description(f"❌ 失敗: {description}")
                logger.error(f"步驟 {i+1} 失敗，流程中止")
                return False
            
            # 更新進度條
            current_progress += weight
            pbar.update(weight)
            pbar.set_description(f"✓ 完成: {description}")
            time.sleep(0.5)  # 讓用戶看到完成狀態
        
        # 完成
        pbar.set_description("🎉 所有步驟完成")
    
    total_elapsed = time.time() - overall_start_time
    
    print("\n" + "=" * 80)
    print("                        🎉 流程執行完成！")
    print("=" * 80)
    print(f"總耗時: {total_elapsed/60:.1f} 分鐘")
    print("📊 您現在可以：")
    print("   1. 查看 AI_Web_Dashboard 進行交易分析")
    print("   2. 檢視 runs/ 目錄下的回測結果")
    print("   3. 查看訓練好的模型檔案 (*.joblib)")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷執行")
    except Exception as e:
        print(f"\n\n💥 發生未預期錯誤: {e}")
        logger.error(f"主程式錯誤: {e}") 