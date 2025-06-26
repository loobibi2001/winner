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
import time
import logging
from datetime import datetime

# 設定路徑
BASE_PATH = r"D:\飆股篩選\winner"
SCRIPTS_DIR = os.path.join(BASE_PATH, "scripts")

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name, description):
    """執行腳本並記錄結果"""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    
    if not os.path.exists(script_path):
        logger.error(f"腳本不存在: {script_path}")
        return False
    
    logger.info(f"開始執行: {description}")
    logger.info(f"腳本路徑: {script_path}")
    
    try:
        # 使用subprocess執行腳本
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=BASE_PATH)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} 執行成功")
            if result.stdout:
                logger.info(f"輸出: {result.stdout}")
            return True
        else:
            logger.error(f"❌ {description} 執行失敗")
            logger.error(f"錯誤: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 執行 {description} 時發生錯誤: {e}")
        return False

def main():
    """主函數"""
    logger.info("===== 一鍵自動化訓練與回測開始 =====")
    start_time = time.time()
    
    # 確保scripts目錄存在
    if not os.path.exists(SCRIPTS_DIR):
        logger.error(f"Scripts目錄不存在: {SCRIPTS_DIR}")
        return
    
    # 步驟1: 更新資料
    logger.info("\n📊 步驟1: 更新股票資料")
    if not run_script("update_all_data.py", "資料更新"):
        logger.warning("資料更新失敗，但繼續執行後續步驟")
    
    # 步驟2: 詢問是否進行超參數調優
    print("\n" + "="*50)
    print("🤖 模型訓練選項:")
    print("1. 標準訓練 (快速)")
    print("2. 超參數調優 (較慢但效果更好)")
    print("="*50)
    
    while True:
        choice = input("請選擇訓練方式 (1 或 2): ").strip()
        if choice in ['1', '2']:
            break
        print("請輸入 1 或 2")
    
    # 步驟2: 訓練模型
    logger.info("\n🤖 步驟2: 訓練模型")
    if choice == '1':
        # 標準訓練
        if not run_script("train_standard.py", "標準模型訓練"):
            logger.error("模型訓練失敗，停止執行")
            return
    else:
        # 超參數調優
        if not run_script("auto_hyperopt.py", "超參數調優訓練"):
            logger.error("超參數調優失敗，停止執行")
            return
    
    # 步驟3: 執行回測
    logger.info("\n📈 步驟3: 執行回測")
    if not run_script("backtest_auto.py", "自動回測"):
        logger.error("回測執行失敗")
        return
    
    # 步驟4: 啟動Web Dashboard
    logger.info("\n🌐 步驟4: 啟動Web Dashboard")
    dashboard_path = os.path.join(BASE_PATH, "AI_Web_Dashboard.py")
    if os.path.exists(dashboard_path):
        logger.info("正在啟動Web Dashboard...")
        try:
            # 啟動dashboard (不等待完成，讓它在背景運行)
            subprocess.Popen([sys.executable, dashboard_path], cwd=BASE_PATH)
            logger.info("✅ Web Dashboard 已啟動")
            logger.info("🌐 請在瀏覽器中開啟: http://localhost:8501")
        except Exception as e:
            logger.error(f"❌ 啟動Web Dashboard失敗: {e}")
    else:
        logger.warning(f"Web Dashboard檔案不存在: {dashboard_path}")
    
    # 完成
    total_time = time.time() - start_time
    logger.info(f"\n🎉 一鍵自動化完成！總耗時: {total_time/60:.2f} 分鐘")
    logger.info("📁 所有結果已保存在winner目錄中")
    logger.info("📊 可以查看以下檔案:")
    logger.info("   - 回測結果: runs/ 目錄")
    logger.info("   - 模型檔案: *.joblib")
    logger.info("   - 調優報告: *_optimization_report.json (如果選擇超參數調優)")

if __name__ == "__main__":
    main() 