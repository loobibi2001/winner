import subprocess
import sys
import time
import os

# 設定基礎路徑
BASE_PATH = r"D:\飆股篩選\winner"

def run_script(script):
    print(f"[開始] 執行 {script} ...")
    start_time = time.time()
    
    # 構建完整腳本路徑
    if not os.path.isabs(script):
        full_script_path = os.path.join(BASE_PATH, script)
    else:
        full_script_path = script
    
    print(f"[路徑] 執行腳本: {full_script_path}")
    
    try:
        result = subprocess.run([sys.executable, full_script_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=BASE_PATH,  # 設定工作目錄
                              timeout=1800)  # 30分鐘timeout
        
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"[完成] {script} 執行完成！耗時: {elapsed_time/60:.1f} 分鐘")
            if result.stdout:
                print(f"[輸出] 輸出摘要: {result.stdout[-500:]}...")  # 只顯示最後500字元
        else:
            elapsed_time = time.time() - start_time
            print(f"[失敗] {script} 執行失敗！耗時: {elapsed_time/60:.1f} 分鐘")
            print(f"錯誤訊息: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"[超時] {script} 執行超時（超過30分鐘），已強制停止")
    except Exception as e:
        print(f"[錯誤] 執行 {script} 時發生未預期錯誤: {e}")

if __name__ == "__main__":
    print("開始執行股價資料更新腳本...")
    print("=" * 60)
    
    # 只更新股價資料（技術指標為主）
    run_script("update_stocks_daily.py")
    print("-" * 40)
    
    # 暫時註解籌碼版資料更新
    # run_script("update_data(籌碼版).py")
    # print("-" * 40)
    
    print("股價資料更新完畢！（籌碼資料暫時停用）") 