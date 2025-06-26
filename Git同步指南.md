# 🔄 Git 同步指南

## 📋 已完成設定
- ✅ Git 倉庫已初始化
- ✅ .gitignore 已設定 (排除日誌、暫存檔等)
- ✅ requirements.txt 已建立 (套件清單)
- ✅ 初始版本已提交

## 🚀 設定遠端倉庫 (選擇一個)

### 方案1：GitHub (推薦)
1. 在 GitHub 上創建新倉庫
2. 複製倉庫網址 (如: https://github.com/你的用戶名/winner.git)
3. 執行以下命令：
```bash
git remote add origin https://github.com/你的用戶名/winner.git
git branch -M main
git push -u origin main
```

### 方案2：GitLab
1. 在 GitLab 上創建新專案
2. 複製專案網址
3. 執行類似命令

### 方案3：本地同步 (如果不用雲端)
```bash
# 在另一台電腦上
git clone 檔案路徑
```

## 📱 在筆電上同步

### 第一次設定
```bash
# 1. 複製專案
git clone https://github.com/你的用戶名/winner.git
cd winner

# 2. 安裝套件
pip install -r requirements.txt

# 3. 檢查環境
python scripts/check_data_integrity.py
```

### 日常同步
```bash
# 在桌面電腦上更新後
git add .
git commit -m "更新說明"
git push

# 在筆電上同步
git pull
```

## 🔄 常用 Git 命令

### 查看狀態
```bash
git status
```

### 查看變更
```bash
git diff
```

### 查看歷史
```bash
git log --oneline
```

### 切換版本
```bash
git checkout 版本號
```

## ⚠️ 重要提醒

### 會同步的檔案
- ✅ 所有 Python 腳本
- ✅ 設定檔案 (.gitignore, requirements.txt)
- ✅ 說明文件
- ✅ scripts/ 目錄

### 不會同步的檔案 (被 .gitignore 排除)
- ❌ 日誌檔案 (*.log)
- ❌ 暫存檔案 (*.tmp, *.bak)
- ❌ 自動產生的報告
- ❌ 大檔案 (可選)

### 資料同步策略
**選項1**: 同步股票資料 (StockData_Parquet/)
- 優點：完全同步
- 缺點：檔案很大

**選項2**: 不同步股票資料
- 優點：同步快速
- 缺點：需要在每台電腦上重新下載資料

## 🎯 建議工作流程

1. **桌面電腦**：開發和測試
2. **筆電**：展示和輕量使用
3. **定期同步**：每天或每次重要更新後

## 📞 遇到問題

### 同步衝突
```bash
git status  # 查看衝突
git merge --abort  # 取消合併
git pull --rebase  # 重新同步
```

### 忘記提交
```bash
git stash  # 暫存變更
git pull   # 同步
git stash pop  # 恢復變更
```

---
**記住**: 每次重要更新後都要 `git add .` → `git commit -m "說明"` → `git push` 