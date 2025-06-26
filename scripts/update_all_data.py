import subprocess
import sys

def run_script(script):
    print(f"執行 {script} ...")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode == 0:
        print(f"{script} 執行完成！")
    else:
        print(f"{script} 執行失敗，請檢查錯誤訊息。")

if __name__ == "__main__":
    run_script("update_stocks_daily.py")
    run_script("update_data(籌碼版).py")
    print("所有資料更新腳本已執行完畢！") 