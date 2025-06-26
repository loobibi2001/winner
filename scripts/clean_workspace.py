# =====================================================================
# 專案: 機器學習策略專案
# 階段: Workspace 自動清理腳本
#
# 功能: 列出可疑用不到的檔案，讓用戶勾選確認後批次刪除
# =====================================================================

import os
import glob
from datetime import datetime

BASE_PATH = r"D:\飆股篩選\winner"
CLEANUP_LIST_PATH = os.path.join(BASE_PATH, "files_to_cleanup.txt")

def find_candidate_files():
    """找出所有可疑用不到的檔案"""
    candidates = []
    
    # 定義可疑檔案模式
    patterns = [
        "*.log",           # 日誌檔案
        "*.tmp",           # 暫存檔案
        "*.bak",           # 備份檔案
        ".DS_Store",       # Mac系統檔案
        "*.txt",           # 文字檔案（除了重要設定檔）
        "*.json",          # JSON檔案（除了重要設定檔）
        "*.png",           # 圖片檔案
        "*.jpg",           # 圖片檔案
        "*.jpeg",          # 圖片檔案
        "*.joblib",        # 模型檔案（除了最新的）
    ]
    
    # 排除的重要檔案
    exclude_files = {
        "stock_list.txt",
        "requirements.txt",
        "README.md",
        "data_integrity_report.txt",
        "交易助手.bat",           # 交易助手主程式
        "快速啟動儀表板.bat",      # 快速啟動儀表板
    }
    
    # 排除的檔案類型
    exclude_extensions = {
        ".bat",                   # 所有批次檔案
        ".py"                     # Python 腳本檔案
    }
    
    exclude_dirs = {
        "scripts",
        "StockData_Parquet"
    }
    
    print("🔍 正在掃描可疑檔案...")
    
    for pattern in patterns:
        # 使用glob找出所有符合模式的檔案
        for file_path in glob.glob(os.path.join(BASE_PATH, "**", pattern), recursive=True):
            # 取得相對路徑
            rel_path = os.path.relpath(file_path, BASE_PATH)
            
            # 檢查是否在排除目錄中
            skip = False
            for exclude_dir in exclude_dirs:
                if exclude_dir in rel_path.split(os.sep):
                    skip = True
                    break
            
            if skip:
                continue
            
            # 檢查是否為排除檔案
            filename = os.path.basename(file_path)
            if filename in exclude_files or any(filename.endswith(ext) for ext in exclude_extensions):
                continue
            
            # 取得檔案大小
            try:
                size = os.path.getsize(file_path)
                size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                candidates.append(f"{rel_path} ({size_str})")
            except:
                candidates.append(f"{rel_path} (無法取得大小)")
    
    return candidates

def save_cleanup_list(candidates):
    """儲存清理清單"""
    with open(CLEANUP_LIST_PATH, 'w', encoding='utf-8') as f:
        f.write("# 可疑用不到的檔案清單\n")
        f.write("# 請在要刪除的檔案前加上 [DELETE]，保留的檔案前加上 [KEEP]\n")
        f.write("# 範例:\n")
        f.write("# [DELETE] model_training.log (15.2KB)\n")
        f.write("# [KEEP] important_config.json (2.1KB)\n")
        f.write(f"# 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# " + "="*50 + "\n\n")
        
        for candidate in candidates:
            f.write(f"[KEEP] {candidate}\n")
    
    print(f"📝 清理清單已儲存至: {CLEANUP_LIST_PATH}")
    print(f"📋 共找到 {len(candidates)} 個可疑檔案")
    print("\n請編輯該檔案，在要刪除的檔案前加上 [DELETE]")

def read_cleanup_decision():
    """讀取用戶的清理決定"""
    if not os.path.exists(CLEANUP_LIST_PATH):
        print("❌ 找不到清理清單檔案")
        return []
    
    to_delete = []
    
    with open(CLEANUP_LIST_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("[DELETE]"):
                # 提取檔案路徑
                file_path = line.replace("[DELETE]", "").strip()
                # 移除大小資訊
                if " (" in file_path:
                    file_path = file_path.split(" (")[0]
                
                full_path = os.path.join(BASE_PATH, file_path)
                if os.path.exists(full_path):
                    to_delete.append(full_path)
    
    return to_delete

def execute_cleanup(to_delete):
    """執行清理"""
    if not to_delete:
        print("✅ 沒有檔案需要刪除")
        return
    
    print(f"\n🗑️ 準備刪除 {len(to_delete)} 個檔案:")
    for file_path in to_delete:
        print(f"  - {os.path.relpath(file_path, BASE_PATH)}")
    
    confirm = input(f"\n❓ 確定要刪除這些檔案嗎？(y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 取消刪除")
        return
    
    deleted_count = 0
    failed_count = 0
    
    for file_path in to_delete:
        try:
            os.remove(file_path)
            print(f"✅ 已刪除: {os.path.relpath(file_path, BASE_PATH)}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ 刪除失敗: {os.path.relpath(file_path, BASE_PATH)} - {e}")
            failed_count += 1
    
    print(f"\n🎉 清理完成！")
    print(f"✅ 成功刪除: {deleted_count} 個檔案")
    if failed_count > 0:
        print(f"❌ 刪除失敗: {failed_count} 個檔案")

def main():
    """主函數"""
    print("🧹 Workspace 自動清理工具")
    print("="*50)
    
    # 步驟1: 找出可疑檔案
    candidates = find_candidate_files()
    
    if not candidates:
        print("✅ 沒有找到可疑檔案，workspace 很乾淨！")
        return
    
    # 步驟2: 儲存清理清單
    save_cleanup_list(candidates)
    
    # 步驟3: 等待用戶編輯
    print("\n📝 請用編輯器開啟以下檔案:")
    print(f"   {CLEANUP_LIST_PATH}")
    print("\n在要刪除的檔案前加上 [DELETE]，保留的檔案前加上 [KEEP]")
    
    input("\n按 Enter 繼續...")
    
    # 步驟4: 讀取用戶決定並執行清理
    to_delete = read_cleanup_decision()
    execute_cleanup(to_delete)

if __name__ == "__main__":
    main() 