# 檔名: update_data.py (真相版)
# =====================================================================
# 本次更新:
#   - 核心修正: 根據偵錯結果，使用 FinMind API 正確的原始欄位名稱。
#   - 標準化: 在儲存前，將所有欄位統一為我們內部使用的標準名稱 (如 volume)。
# =====================================================================
import os
import pandas as pd
import time
import logging
import tqdm
import numpy as np
from datetime import datetime, timedelta
from FinMind.data import FinMindApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FINMIND_API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wNi0yMyAxNTo1ODoxMyIsInVzZXJfaWQiOiJsb29iaWJpMjAwMSIsImlwIjoiMTE0LjQ3LjE0Ni45OCJ9.VtbIGVAnRXcTEUsd1T87ryIn1dkCON-f8rfwhNgih0k" 
BASE_DIR = r"D:\飆股篩選\winner"
DATA_DIR = os.path.join(BASE_DIR, "StockData_Parquet")
PRICE_DIR = os.path.join(DATA_DIR, "Price")
INVESTOR_DIR = os.path.join(DATA_DIR, "InstitutionalInvestors")
HOLDER_DIR = os.path.join(DATA_DIR, "Shareholding")
LIST_PATH = os.path.join(BASE_DIR, "stock_list.txt")
MARKET_INDEX_ID = "TAIEX"
DOWNLOAD_START_DATE = "1900-01-01"

def download_data(api: FinMindApi, dataset: str, stock_id: str):
    start_date = DOWNLOAD_START_DATE
    end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        df = api.get_data(dataset=dataset, data_id=stock_id, start_date=start_date, end_date=end_date)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception as e:
        logger.error(f"下載 {stock_id} 的 {dataset} 時發生錯誤: {e}")
        return pd.DataFrame()

def main():
    if FINMIND_API_TOKEN == "YOUR_FINMIND_API_TOKEN":
        logger.error("錯誤：請在腳本中填寫您自己的 FinMind API Token！")
        return

    os.makedirs(PRICE_DIR, exist_ok=True); os.makedirs(INVESTOR_DIR, exist_ok=True); os.makedirs(HOLDER_DIR, exist_ok=True)
    
    api = FinMindApi()
    api.login_by_token(api_token=FINMIND_API_TOKEN)
    
    with open(LIST_PATH, "r", encoding="utf-8") as f:
        stock_list = sorted([l.strip() for l in f if l.strip().isdigit()])
    if MARKET_INDEX_ID not in stock_list:
        stock_list.append(MARKET_INDEX_ID)
        
    logger.info(f"準備開始更新總共 {len(stock_list)} 筆數據...")

    for stock_id in tqdm.tqdm(stock_list, desc="更新數據進度"):
        time.sleep(0.2)
        # 1. 處理股價數據
        df_price = download_data(api, "TaiwanStockPrice", stock_id)
        if not df_price.empty:
            df_price = df_price.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'})
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            df_price = df_price.reindex(columns=required_cols)
            df_price['date'] = pd.to_datetime(df_price['date'])
            df_price['volume'] = pd.to_numeric(df_price['volume'], errors='coerce').fillna(0).astype(np.int64)
            df_price = df_price.set_index('date')
            df_price.to_parquet(os.path.join(PRICE_DIR, f"{stock_id}_history.parquet"))

        # 2. 處理法人買賣超數據
        df_investor = download_data(api, "TaiwanStockInstitutionalInvestorsBuySell", stock_id)
        if not df_investor.empty:
            df_investor['date'] = pd.to_datetime(df_investor['date'])
            df_investor.set_index('date', inplace=True)
            df_investor.to_parquet(os.path.join(INVESTOR_DIR, f"{stock_id}.parquet"))
        
        # 3. 處理股權分散數據
        df_holder = download_data(api, "TaiwanStockShareholding", stock_id)
        if not df_holder.empty:
            df_holder['date'] = pd.to_datetime(df_holder['date'])
            df_holder.set_index('date', inplace=True)
            df_holder.to_parquet(os.path.join(HOLDER_DIR, f"{stock_id}.parquet"))
            
    logger.info("\n數據更新任務完成！")

if __name__ == '__main__':
    main()