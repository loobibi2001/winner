# Winner 專案檔案說明

## 📁 主要目錄結構

### 🎯 核心腳本 (根目錄)
- **AI_Web_Dashboard.py** - AI量化交易Web介面，用Streamlit建立的視覺化儀表板
- **V40.1_XGBoost模型訓練腳本_修正版.py** - XGBoost模型訓練主腳本，訓練股票多空預測模型
- **V40.1_XGBoost回測最終版.py** - 回測主腳本，測試模型在歷史數據上的表現
- **V45_Walk_Forward_Framework.py** - Walk-Forward分析框架，用於驗證策略穩健性
- **update_stocks_daily.py** - 每日股票資料更新腳本
- **update_data(籌碼版).py** - 籌碼面資料更新腳本

### 📂 scripts/ 目錄 (自動化工具)
- **update_all_data.py** - 一鍵更新所有資料腳本
- **train_and_backtest.py** - 一鍵訓練與回測腳本 (含超參數調優選項)
- **clean_workspace.py** - workspace清理工具，刪除用不到的檔案
- **check_data_integrity.py** - 資料完整性檢查工具
- **auto_hyperopt.py** - 完整版超參數自動調優腳本 (需要talib)
- **simple_hyperopt.py** - 簡化版超參數調優腳本 (不需要talib)
- **test_hyperopt.py** - 超參數調優測試腳本
- **train_standard.py** - 標準模型訓練腳本 (固定參數)
- **backtest_auto.py** - 自動回測腳本

### 📊 資料目錄
- **StockData_Parquet/** - 股票歷史資料庫 (parquet格式)
  - 包含所有股票的開高低收量資料
  - 每個檔案格式: `股票代碼_history.parquet`
- **InstitutionalInvestors/** - 法人投資資料
- **Price/** - 價格資料
- **Shareholding/** - 持股資料

### 🎯 模型檔案
- **xgboost_long_short_model.joblib** - 訓練好的XGBoost多空預測模型
- **regime_model.joblib** - 市場情勢判斷模型
- **random_forest_model.joblib** - 隨機森林模型 (備用)

### 📈 回測結果
- **runs/** - 回測結果目錄
  - 包含每次回測的KPI報告和圖表
- **charts_final_scenario_RSI60/** - RSI策略回測圖表

### 📋 設定檔案
- **stock_list.txt** - 股票代碼清單 (自動產生)
- **data_integrity_report.txt** - 資料完整性檢查報告
- **files_to_cleanup.txt** - 清理檔案清單 (自動產生)

### 📊 報告檔案
- **optimization_summary.json** - 超參數調優總結報告
- **stock_model_optimization_report.json** - 股票模型調優報告
- **market_model_optimization_report.json** - 市場模型調優報告

## 🚀 常用工作流程

### 1. 日常使用流程
```bash
# 更新資料
python scripts/update_all_data.py

# 訓練模型 (選擇標準或調優)
python scripts/train_and_backtest.py

# 啟動Web Dashboard
python AI_Web_Dashboard.py
```

### 2. 資料檢查流程
```bash
# 檢查資料完整性
python scripts/check_data_integrity.py

# 清理workspace
python scripts/clean_workspace.py
```

### 3. 模型調優流程
```bash
# 簡化版調優 (推薦)
python scripts/simple_hyperopt.py

# 完整版調優 (需要talib)
python scripts/auto_hyperopt.py
```

## 🔧 重要提醒

### 必須保留的檔案
- **scripts/** 整個目錄
- **StockData_Parquet/** 整個目錄
- **AI_Web_Dashboard.py**
- **V40.1_XGBoost模型訓練腳本_修正版.py**
- **V40.1_XGBoost回測最終版.py**
- **stock_list.txt**

### 可以刪除的檔案
- **.log** 檔案 (日誌)
- **.tmp** 檔案 (暫存)
- **舊的回測結果** (runs/ 目錄下的舊資料)
- **舊的圖表** (charts_final_scenario_RSI60/ 下的舊圖)

### 環境需求
```bash
pip install pandas numpy scikit-learn xgboost optuna pyarrow streamlit
```

## 📝 檔案更新時間
- 最後更新: 2025-06-26
- 版本: v1.0 