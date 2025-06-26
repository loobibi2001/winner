# 檔名: V40.1_XGBoost模型訓練腳本.py (語法修正與測試版)
# =====================================================================
# 專案: 機器學習策略專案
# 階段: 模型訓練 - XGBoost 多空雙向模型與市場情勢模型
#
# 本次更新:
#   - 修正 calculate_features 函數中的語法錯誤。
#   - 保留檔案完整性測試的 print 語句。
#   - 修正 XGBoost callbacks 相容性問題。
# =====================================================================

import os
import joblib
import time
import logging
import numpy as np
import pandas as pd
import talib
import xgboost as xgb
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import defaultdict
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# --- 全域設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='model_training.log', filemode='a')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger = logging.getLogger('')
logger.addHandler(console_handler)

BASE_DIR = r"D:\飆股篩選\winner"
DATA_DIR = os.path.join(BASE_DIR, "StockData_Parquet")
LIST_PATH = os.path.join(BASE_DIR, "stock_list.txt")
STOCK_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "xgboost_long_short_model.joblib")
REGIME_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "regime_model.joblib")
MARKET_INDEX_ID = "TAIEX"
TRAIN_START_DATE = "2010-01-01"
TRAIN_END_DATE = "2023-12-31"
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
FUTURE_DAYS_FOR_STOCK_LABEL = 5
STOCK_BUY_THRESHOLD = 0.03
STOCK_SELL_THRESHOLD = -0.03
FUTURE_DAYS_FOR_MARKET_LABEL = 20
MARKET_BULL_THRESHOLD = 0.05
MARKET_BEAR_THRESHOLD = -0.03

# --- 核心函數 ---
def prep_data(stock_id: str) -> Optional[pd.DataFrame]:
    file_path = os.path.join(DATA_DIR, f"{stock_id}_history.parquet")
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_parquet(file_path)
        df.columns = [c.lower() for c in df.columns]
        if 'date' not in df.columns: df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').set_index('date')
        required = ['high', 'low', 'close', 'open', 'volume']
        if any(c not in df.columns for c in required): return None
        if df[required].isnull().any().any(): return None
        if (df['volume'] <= 0).any(): return None
        df['next_day_open'] = df['open'].shift(-1)
        return df.dropna(subset=required)
    except Exception as e:
        logger.debug(f"讀取 {stock_id} 數據時出錯: {e}")
        return None

def calculate_features(df: pd.DataFrame, is_market: bool = False) -> pd.DataFrame:
    """
    計算股票或市場指數的技術分析特徵。
    """
    df_feat = df.copy()
    # 定義市場和股票的特徵列表，確保訓練和預測時使用相同的特徵集合
    market_feature_list = ['price_change_1d','price_change_5d','ma_20','ma20_slope','ma_60','rsi_14','macd','kd_k','atr_14','bollinger_width']
    stock_feature_list = market_feature_list + ['volume_change_1d', 'price_vs_ma20', 'williams_r', 'ma_5', 'macdsignal', 'macdhist', 'kd_d']

    df_feat['price_change_1d'] = df_feat['close'].pct_change(1)
    df_feat['price_change_5d'] = df_feat['close'].pct_change(5)
    df_feat['volume_change_1d'] = df_feat['volume'].pct_change(1)
    
    # 確保數據類型正確並轉換為numpy array
    close_array = df_feat['close'].astype(float).to_numpy()
    high_array = df_feat['high'].astype(float).to_numpy()
    low_array = df_feat['low'].astype(float).to_numpy()
    
    df_feat['ma_5'] = talib.SMA(close_array, 5)
    df_feat['ma_20'] = talib.SMA(close_array, 20)
    
    # 加入ma20_slope特徵計算
    df_feat['ma20_slope'] = df_feat['ma_20'].pct_change(1)
    
    df_feat['ma_60'] = talib.SMA(close_array, 60)
    df_feat['rsi_14'] = talib.RSI(close_array, 14)
    df_feat['macd'], df_feat['macdsignal'], df_feat['macdhist'] = talib.MACD(close_array, 12, 26, 9)
    df_feat['kd_k'], df_feat['kd_d'] = talib.STOCH(high_array, low_array, close_array, 9, 3, 3)
    df_feat['atr_14'] = talib.ATR(high_array, low_array, close_array, 14)
    upper, middle, lower = talib.BBANDS(close_array, 20)
    df_feat['bollinger_width'] = np.where(middle > 0, (upper - lower) / middle, 0)

    if not is_market:
        df_feat['price_vs_ma20'] = np.where(df_feat['ma_20'] > 0, df_feat['close'] / df_feat['ma_20'], 1)
        df_feat['williams_r'] = talib.WILLR(high_array, low_array, close_array, 14)

    # 在訓練數據生成時，先不急著dropna，確保標籤計算有足夠數據。
    # 最終的dropna會在合併特徵和標籤後進行。
    return df_feat.replace([np.inf, -np.inf], np.nan)

def create_stock_labels(df: pd.DataFrame, future_days: int, buy_threshold: float, sell_threshold: float) -> pd.Series:
    if len(df) <= future_days: return pd.Series(dtype=int)
    df['future_open'] = df['open'].shift(-future_days)
    df['future_return'] = (df['future_open'] - df['next_day_open']) / df['next_day_open']
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[df['future_return'] >= buy_threshold] = 1
    labels[df['future_return'] <= sell_threshold] = -1
    return labels.dropna()

def create_market_labels(df: pd.DataFrame, future_days: int, bull_threshold: float, bear_threshold: float) -> pd.Series:
    if len(df) <= future_days: return pd.Series(dtype=int)
    df['future_close'] = df['close'].shift(-future_days)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[df['future_return'] >= bull_threshold] = 1
    return labels.dropna()

def train_models():
    logger.info("===== XGBoost 模型訓練開始 =====")
    try:
        with open(LIST_PATH, "r", encoding="utf-8") as f:
            stock_list = sorted([l.strip() for l in f if l.strip().isdigit()])
    except FileNotFoundError: logger.error(f"錯誤：找不到股票列表檔案 {LIST_PATH}！"); return

    all_stock_combined_data = []
    stock_model_feature_cols = ['price_change_1d','price_change_5d','volume_change_1d','ma_5','ma_20','ma_60','rsi_14','macd','macdsignal','macdhist','kd_k','kd_d','atr_14','bollinger_width','price_vs_ma20','williams_r']

    logger.info("加載所有個股數據並計算特徵及標籤...")
    for sid in tqdm.tqdm(stock_list, desc="處理個股數據"):
        df_raw = prep_data(sid)
        if df_raw is None: continue
        df_feat = calculate_features(df_raw, is_market=False)
        labels = create_stock_labels(df_raw, FUTURE_DAYS_FOR_STOCK_LABEL, STOCK_BUY_THRESHOLD, STOCK_SELL_THRESHOLD)
        combined_df = df_feat.merge(labels.rename('label'), left_index=True, right_index=True)
        combined_df = combined_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
        combined_df = combined_df.dropna(subset=stock_model_feature_cols + ['label'])
        if not combined_df.empty: all_stock_combined_data.append(combined_df)

    if not all_stock_combined_data: logger.warning("沒有足夠的股票數據來訓練個股模型。"); return
    
    stock_train_df = pd.concat(all_stock_combined_data)
    X_stock = stock_train_df[stock_model_feature_cols]
    y_stock = stock_train_df['label']
    stock_label_encoder = LabelEncoder()
    y_stock_encoded = stock_label_encoder.fit_transform(y_stock)
    logger.info(f"股票模型原始標籤映射: {list(stock_label_encoder.classes_)} -> {list(range(len(stock_label_encoder.classes_)))}")
    logger.info(f"股票模型訓練數據量: {len(X_stock)} 筆"); logger.info(f"股票模型標籤分佈: \n{y_stock.value_counts()}")

    if not X_stock.empty and len(np.unique(y_stock_encoded)) > 1:
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_stock, y_stock_encoded, test_size=TEST_SIZE_RATIO, random_state=RANDOM_STATE, stratify=y_stock_encoded)
        logger.info("開始訓練個股多空預測模型...")
        stock_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(stock_label_encoder.classes_), eval_metric='mlogloss', use_label_encoder=False, n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.7, colsample_bytree=0.7, random_state=RANDOM_STATE, n_jobs=-1)
        
        # 【檔案完整性測試】
        print("\n<<<<< 正在執行修改後的最新檔案，即將進入 stock_model.fit >>>>>\n")

        # 移除 early_stopping_rounds 參數以確保相容性
        stock_model.fit(X_train_s, y_train_s, eval_set=[(X_test_s, y_test_s)], verbose=False)

        y_pred_s = stock_model.predict(X_test_s)
        logger.info(f"股票模型測試集準確度: {accuracy_score(y_test_s, y_pred_s):.4f}")
        logger.info(f"股票模型分類報告:\n{classification_report(y_test_s, y_pred_s, target_names=stock_label_encoder.classes_.astype(str))}")
        joblib.dump({'model': stock_model, 'label_encoder': stock_label_encoder}, STOCK_MODEL_SAVE_PATH)
        logger.info(f"個股多空預測模型已儲存至: {STOCK_MODEL_SAVE_PATH}")
    else: logger.warning("沒有足夠或多樣的股票數據來訓練個股模型。")

    market_raw_data = prep_data(MARKET_INDEX_ID)
    if market_raw_data is None: logger.error("無法加載大盤數據，無法訓練市場情勢模型。"); return
    
    market_features = calculate_features(market_raw_data, is_market=True)
    market_labels = create_market_labels(market_raw_data, FUTURE_DAYS_FOR_MARKET_LABEL, MARKET_BULL_THRESHOLD, MARKET_BEAR_THRESHOLD)
    combined_market_df = market_features.merge(market_labels.rename('label'), left_index=True, right_index=True)
    combined_market_df = combined_market_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    market_model_feature_cols = ['price_change_1d','price_change_5d','ma_20','ma20_slope','ma_60','rsi_14','macd','kd_k','atr_14','bollinger_width']
    combined_market_df = combined_market_df.dropna(subset=market_model_feature_cols + ['label'])
    X_market = combined_market_df[market_model_feature_cols]
    y_market = combined_market_df['label']

    logger.info(f"市場情勢模型訓練數據量: {len(X_market)} 筆"); logger.info(f"市場情勢模型標籤分佈: \n{y_market.value_counts()}")

    if not X_market.empty and len(np.unique(y_market)) > 1:
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_market, y_market, test_size=TEST_SIZE_RATIO, random_state=RANDOM_STATE, stratify=y_market)
        logger.info("開始訓練市場情勢判斷模型...")
        regime_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)

        # 移除 early_stopping_rounds 參數以確保相容性
        regime_model.fit(X_train_m, y_train_m, eval_set=[(X_test_m, y_test_m)], verbose=False)

        y_pred_m = regime_model.predict(X_test_m)
        logger.info(f"市場情勢模型測試集準確度: {accuracy_score(y_test_m, y_pred_m):.4f}")
        logger.info(f"市場情勢模型分類報告:\n{classification_report(y_test_m, y_pred_m, target_names=['bear', 'bull'])}")
        joblib.dump(regime_model, REGIME_MODEL_SAVE_PATH)
        logger.info(f"市場情勢判斷模型已儲存至: {REGIME_MODEL_SAVE_PATH}")
    else: logger.warning("沒有足夠或多樣的大盤數據來訓練市場情勢模型。")

    logger.info("===== XGBoost 模型訓練完成 =====")

if __name__ == '__main__':
    train_models()