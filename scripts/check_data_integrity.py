# =====================================================================
# 專案: 機器學習策略專案
# 階段: 股票資料完整性自動檢查腳本
#
# 功能: 檢查每個parquet檔案是否有100筆以上資料，且open/high/low/close/volume欄位無缺失
#      輸出完整性報告，方便自動化流程前檢查
# =====================================================================

import os
import pandas as pd

BASE_PATH = r"D:\飆股篩選\winner"
DATA_DIR = os.path.join(BASE_PATH, "StockData_Parquet")
REPORT_PATH = os.path.join(BASE_PATH, "data_integrity_report.txt")

REQUIRED_COLS = ['open', 'high', 'low', 'close', 'volume']
MIN_ROWS = 100

results = []

for fname in os.listdir(DATA_DIR):
    if not fname.endswith('_history.parquet'):
        continue
    stock_id = fname.replace('_history.parquet', '')
    fpath = os.path.join(DATA_DIR, fname)
    try:
        df = pd.read_parquet(fpath)
        df.columns = [c.lower() for c in df.columns]
        row_count = len(df)
        missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
        has_nan = df[REQUIRED_COLS].isnull().any().any() if not missing_cols else True
        if row_count < MIN_ROWS:
            status = f'❌ 少於{MIN_ROWS}筆 ({row_count})'
        elif missing_cols:
            status = f'❌ 缺少欄位: {missing_cols}'
        elif has_nan:
            status = f'❌ 有NaN缺失值'
        else:
            status = '✅ 完整'
        results.append(f'{stock_id}: {status}')
    except Exception as e:
        results.append(f'{stock_id}: ❌ 讀取錯誤: {e}')

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print(f'檢查完成，報告已儲存至: {REPORT_PATH}')
print('\n'.join(results[:20]))  # 顯示前20筆結果 