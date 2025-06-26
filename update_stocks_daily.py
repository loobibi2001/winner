from FinMind.data import DataLoader
import os
import pandas as pd
from datetime import date, timedelta, datetime
import time
import multiprocessing as mp
from functools import partial

def process_stock(stock_id):
    # 直接在函數內部定義必要參數，避免多進程下 globals() 失效
    from FinMind.data import DataLoader
    import pandas as pd
    import os
    import time
    finmind_api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wNS0yNSAxODo0MzozMSIsInVzZXJfaWQiOiJsb29iaWJpMjAwMSIsImlwIjoiMTE0LjQ3LjE1MS4xNTIifQ.DlJCvlAcFZBsFVuSja0VpVxX1AsH3MdFE-s32HAMfhA"
    stock_folder = r"D:\飆股篩選\winner\StockData_Parquet"
    today = date.today()
    delay_seconds = 0.5
    default_start_date_for_missing = '1900-01-01'
    try:
        api = DataLoader()
        api.login_by_token(api_token=finmind_api_token)
        file_path = os.path.join(stock_folder, f"{stock_id}_history.csv")
        parquet_path = file_path.replace('.csv', '.parquet')
        
        # 讀取現有檔案，決定起始日
        if os.path.exists(file_path):
            try:
                old_df = pd.read_csv(file_path, encoding='utf-8')
                last_date_in_file = old_df['date'].max()
                if isinstance(last_date_in_file, pd.Series):
                    last_date_in_file = last_date_in_file.iloc[0]
                    
                # 檢查是否已經是最新資料（今天或昨天）
                if pd.notnull(last_date_in_file):
                    last_date = pd.to_datetime(last_date_in_file).date()
                    if last_date >= today - timedelta(days=1):  # 如果是今天或昨天，跳過更新
                        return stock_id, "already_latest", f"資料已是最新 ({last_date})", 0
                        
            except Exception:
                last_date_in_file = None
        else:
            last_date_in_file = None
            
        # 決定下載起始日，確保是 str
        if last_date_in_file is not None and pd.notnull(last_date_in_file):
            start_date = str(last_date_in_file)
        else:
            start_date = default_start_date_for_missing
            
        # 下載新資料
        new_df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=str(today))
        if new_df.empty:
            return stock_id, "already_latest", "無新資料", 0
            
        # 欄位對應
        new_df = new_df.rename(columns={
            'date': 'date',
            'open': 'Open',
            'max': 'High',
            'min': 'Low',
            'close': 'Close',
            'Trading_Volume': 'Volume',
            'trading_volume': 'Volume'
        })
        cols_to_save = [col for col in ['date', 'Open', 'High', 'Low', 'Close', 'Volume'] if col in new_df.columns]
        new_df_to_save = new_df[cols_to_save]
        
        # 存檔
        if os.path.exists(file_path) and last_date_in_file is not None:
            new_df_to_save.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8')
            # Parquet 合併
            if os.path.exists(parquet_path):
                old_parquet = pd.read_parquet(parquet_path)
                combined = pd.concat([old_parquet, new_df_to_save], ignore_index=True)
                combined.to_parquet(parquet_path, index=False)
            else:
                new_df_to_save.to_parquet(parquet_path, index=False)
            return stock_id, "updated", f"成功附加新資料 ({len(new_df_to_save)} 筆)", len(new_df_to_save)
        else:
            new_df_to_save.to_csv(file_path, index=False, encoding='utf-8')
            new_df_to_save.to_parquet(parquet_path, index=False)
            return stock_id, "newly_downloaded", f"成功下載新檔案 ({len(new_df_to_save)} 筆)", len(new_df_to_save)
    except Exception as e:
        return stock_id, "error", str(e), 0
    finally:
        time.sleep(delay_seconds)

def main():
    print(f"--- 每日更新腳本開始執行 (v2.4 - 4進程並行下載版) --- ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")

    # --- 設定 (無變化) ---
    global finmind_api_token, base_path, stock_data_folder_name, stock_list_file_name, stock_folder, stock_list_file_path, today, delay_seconds, default_start_date_for_missing
    finmind_api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wNS0yNSAxODo0MzozMSIsInVzZXJfaWQiOiJsb29iaWJpMjAwMSIsImlwIjoiMTE0LjQ3LjE1MS4xNTIifQ.DlJCvlAcFZBsFVuSja0VpVxX1AsH3MdFE-s32HAMfhA"
    base_path = r"D:\飆股篩選"
    stock_data_folder_name = "StockData_Parquet"
    stock_list_file_name = "stock_list.txt"
    stock_folder = r"D:\飆股篩選\winner\StockData_Parquet"
    stock_list_file_path = os.path.join(base_path, stock_list_file_name)
    today = date.today()
    delay_seconds = 0.5
    default_start_date_for_missing = '1900-01-01'

    # --- 登入與資料夾設定 (無變化) ---
    print(f"設定的基礎路徑: {base_path}")
    print(f"設定的 CSV 資料夾路徑: {stock_folder}")
    print(f"將依據此檔案的內容進行處理: {stock_list_file_path}")

    print("--- 登入 FinMind API ---")
    try:
        api = DataLoader()
        api.login_by_token(api_token=finmind_api_token)
        print("--- 成功登入 FinMind API ---")
    except Exception as e:
        print(f"!!!!!! 登入 FinMind API 時發生錯誤: {e} !!!!!!")
        exit()

    os.makedirs(stock_folder, exist_ok=True)

    # --- 讀取股票列表 (無變化) ---
    print(f"--- 正在從 {stock_list_file_path} 讀取唯一的股票/指數代碼列表 ---")
    stock_ids_to_update = []
    try:
        with open(stock_list_file_path, "r", encoding="utf-8") as f:
            stock_ids_to_update = [line.strip() for line in f if line.strip()]
        if not stock_ids_to_update:
            print(f"!!!!!! 錯誤：{stock_list_file_path} 是空的。腳本結束。 !!!!!!")
            exit()
        print(f"成功從 {stock_list_file_path} 讀取了 {len(stock_ids_to_update)} 個代碼進行處理。")
    except Exception as e:
        print(f"!!!!!! 讀取 {stock_list_file_path} 時發生嚴重錯誤: {e} !!!!!!")
        exit()

    # --- 8進程並行下載 ---
    print(f"\n--- 開始使用 8 進程並行檢查並更新/下載 {len(stock_ids_to_update)} 支股票/指數資料 ---")

    # 統計變數
    updated_count, newly_downloaded_count, already_latest_count, error_count = 0, 0, 0, 0

    # 使用 8 進程並行處理
    with mp.Pool(processes=8) as pool:
        results = pool.imap_unordered(process_stock, stock_ids_to_update)
        
        for i, (stock_id, status, message, data_count) in enumerate(results):
            print(f"  > ({i+1}/{len(stock_ids_to_update)}) 正在處理: {stock_id}")
            print(f"    >> {message}")
            
            # 更新統計
            if status == "updated":
                updated_count += 1
            elif status == "newly_downloaded":
                newly_downloaded_count += 1
            elif status == "already_latest":
                already_latest_count += 1
            elif status == "error":
                error_count += 1
            
            # 簡短延遲避免過度頻繁的輸出
            time.sleep(0.1)

    print(f"\n--- 每日更新腳本執行完畢 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print(f"更新統計:")
    print(f"  - 成功更新: {updated_count} 支")
    print(f"  - 新下載: {newly_downloaded_count} 支")
    print(f"  - 已是最新: {already_latest_count} 支")
    print(f"  - 錯誤: {error_count} 支")
    print(f"  - 總計處理: {len(stock_ids_to_update)} 支")

if __name__ == '__main__':
    main()