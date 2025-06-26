# =====================================================================
# 專案: 機器學習策略專案
# 階段: 模型訓練 - XGBoost 多空雙向模型與市場情勢模型 (修正版)
#
# 本次更新:
#   - 修正標籤生成邏輯，確保與實際交易邏輯一致
#   - 解決數據洩漏問題
#   - 統一特徵計算，確保訓練與回測一致性
#   - 優化模型參數
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

BASE_PATH = r"D:\飆股篩選\winner"
WINNER_DIR = BASE_PATH
DATA_DIR = os.path.join(BASE_PATH, "StockData_Parquet")
LIST_PATH = os.path.join(BASE_PATH, "stock_list.txt")
STOCK_MODEL_SAVE_PATH = os.path.join(BASE_PATH, "xgboost_long_short_model.joblib")
REGIME_MODEL_SAVE_PATH = os.path.join(BASE_PATH, "regime_model.joblib")
MARKET_INDEX_ID = "TAIEX"

# --- 訓練參數 ---
TRAIN_START_DATE = "2010-01-01"
TRAIN_END_DATE = "2023-12-31"
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42

# --- 標籤生成參數 ---
FUTURE_DAYS_FOR_STOCK_LABEL = 5
STOCK_BUY_THRESHOLD = 0.03    # 3% 漲幅作為買入信號
STOCK_SELL_THRESHOLD = -0.03  # -3% 跌幅作為賣出信號

FUTURE_DAYS_FOR_MARKET_LABEL = 20
MARKET_BULL_THRESHOLD = 0.05  # 5% 漲幅作為牛市信號

# --- 核心函數 ---
def prep_data(stock_id: str) -> Optional[pd.DataFrame]:
    """
    準備股票數據，避免數據洩漏
    """
    file_path = os.path.join(DATA_DIR, f"{stock_id}_history.parquet")
    if not os.path.exists(file_path): 
        return None
    
    try:
        df = pd.read_parquet(file_path)
        # 確保欄位名稱轉換為小寫
        df.columns = [c.lower() for c in df.columns]
        
        if 'date' not in df.columns: 
            df = df.reset_index()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').set_index('date')
        
        required = ['high', 'low', 'close', 'open', 'volume']
        
        # 檢查必要欄位
        if any(c not in df.columns for c in required): 
            logger.debug(f"{stock_id} 缺少必要欄位: {required}, 實際欄位: {df.columns.tolist()}")
            return None
        
        # 檢查空值
        if df[required].isnull().any().any(): 
            logger.debug(f"{stock_id} 包含空值")
            return None
        
        # 檢查交易量
        if (df['volume'] <= 0).any(): 
            logger.debug(f"{stock_id} 包含零或負交易量")
            return None
        
        # 計算次日開盤價（用於標籤生成，避免數據洩漏）
        df['next_day_open'] = df['open'].shift(-1)
        
        return df.dropna(subset=required)
    except Exception as e:
        logger.debug(f"讀取 {stock_id} 數據時出錯: {e}")
        return None

def calculate_features(df: pd.DataFrame, is_market: bool = False) -> pd.DataFrame:
    """
    計算技術分析特徵，確保與回測腳本一致
    """
    df_feat = df.copy()
    
    # 確保與回測腳本一致的特徵列表
    market_feature_list = ['price_change_1d','price_change_5d','ma_20','ma20_slope','ma_60','rsi_14','macd','kd_k','atr_14','bollinger_width']
    stock_feature_list = market_feature_list + ['volume_change_1d', 'price_vs_ma20', 'williams_r', 'ma_5', 'macdsignal', 'macdhist', 'kd_d']
    
    # 基本價格變化
    df_feat['price_change_1d'] = df_feat['close'].pct_change(1)
    df_feat['price_change_5d'] = df_feat['close'].pct_change(5)
    df_feat['volume_change_1d'] = df_feat['volume'].pct_change(1)
    
    # 確保數據類型正確並轉換為numpy array
    close_array = df_feat['close'].astype(float).to_numpy()
    high_array = df_feat['high'].astype(float).to_numpy()
    low_array = df_feat['low'].astype(float).to_numpy()
    
    # 移動平均線
    df_feat['ma_5'] = talib.SMA(close_array, 5)
    df_feat['ma_20'] = talib.SMA(close_array, 20)
    df_feat['ma_60'] = talib.SMA(close_array, 60)
    
    # 計算20日均線斜率
    df_feat['ma20_slope'] = df_feat['ma_20'].pct_change(1, fill_method=None)
    
    # 技術指標
    df_feat['rsi_14'] = talib.RSI(close_array, 14)
    df_feat['macd'], df_feat['macdsignal'], df_feat['macdhist'] = talib.MACD(close_array, 12, 26, 9)
    df_feat['kd_k'], df_feat['kd_d'] = talib.STOCH(high_array, low_array, close_array, 9, 3, 3)
    df_feat['atr_14'] = talib.ATR(high_array, low_array, close_array, 14)
    
    # 布林通道
    upper, middle, lower = talib.BBANDS(close_array, 20)
    df_feat['bollinger_width'] = np.where(middle > 0, (upper - lower) / middle, 0)
    
    # 股票特有特徵
    if not is_market:
        df_feat['price_vs_ma20'] = np.where(df_feat['ma_20'] > 0, df_feat['close'] / df_feat['ma_20'], 1)
        df_feat['williams_r'] = talib.WILLR(high_array, low_array, close_array, 14)
    
    # 處理無限值和NaN
    return df_feat.replace([np.inf, -np.inf], np.nan)

def create_stock_labels(df: pd.DataFrame, future_days: int, buy_threshold: float, sell_threshold: float) -> pd.Series:
    """
    創建股票交易標籤
    修正：確保標籤生成邏輯與實際交易一致
    """
    if len(df) <= future_days: 
        return pd.Series(dtype=int)
    
    # 使用 next_day_open 作為進場價，future_open 作為出場價
    df['future_open'] = df['open'].shift(-future_days)
    df['future_return'] = (df['future_open'] - df['next_day_open']) / df['next_day_open']
    
    labels = pd.Series(0, index=df.index, dtype=int)  # 0: 持有
    labels[df['future_return'] >= buy_threshold] = 1   # 1: 買入信號
    labels[df['future_return'] <= sell_threshold] = -1 # -1: 賣出信號
    
    return labels.dropna()

def create_market_labels(df: pd.DataFrame, future_days: int, bull_threshold: float) -> pd.Series:
    """
    創建市場情勢標籤
    修正：確保標籤平衡
    """
    if len(df) <= future_days: 
        return pd.Series(dtype=int)
    
    df['future_close'] = df['close'].shift(-future_days)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
    labels = pd.Series(0, index=df.index, dtype=int)  # 0: 熊市
    labels[df['future_return'] >= bull_threshold] = 1  # 1: 牛市
    
    return labels.dropna()

def train_models():
    """
    訓練XGBoost模型
    """
    logger.info("===== XGBoost 模型訓練開始 (修正版) =====")
    
    # 載入股票列表
    try:
        with open(LIST_PATH, "r", encoding="utf-8") as f:
            stock_list = sorted([l.strip() for l in f if l.strip().isdigit()])
    except FileNotFoundError: 
        logger.error(f"錯誤：找不到股票列表檔案 {LIST_PATH}！")
        return
    
    # --- 訓練股票模型 ---
    logger.info("開始訓練股票多空預測模型...")
    
    all_stock_combined_data = []
    stock_model_feature_cols = ['price_change_1d','price_change_5d','volume_change_1d','ma_5','ma_20','ma20_slope','ma_60','rsi_14','macd','macdsignal','macdhist','kd_k','kd_d','atr_14','bollinger_width','price_vs_ma20','williams_r']
    
    logger.info("加載所有個股數據並計算特徵及標籤...")
    successful_stocks = 0
    for sid in tqdm.tqdm(stock_list, desc="處理個股數據"):
        df_raw = prep_data(sid)
        if df_raw is None:
            continue
        
        # 檢查數據長度是否足夠
        if len(df_raw) < 100:  # 至少需要100筆資料來計算技術指標
            continue
            
        df_feat = calculate_features(df_raw, is_market=False)
        labels = create_stock_labels(df_raw, FUTURE_DAYS_FOR_STOCK_LABEL, STOCK_BUY_THRESHOLD, STOCK_SELL_THRESHOLD)
        
        if labels.empty:
            continue
        
        # 合併特徵和標籤
        combined_df = df_feat.merge(labels.rename('label'), left_index=True, right_index=True)
        combined_df = combined_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
        
        # 只針對需要的特徵做dropna
        valid_df = combined_df.dropna(subset=stock_model_feature_cols + ['label'])
        
        if len(valid_df) >= 50:  # 至少需要50筆有效數據
            all_stock_combined_data.append(valid_df)
            successful_stocks += 1
            if successful_stocks <= 5:  # 只顯示前5個成功的股票
                print(f"{sid} 成功: {len(valid_df)} 筆有效數據")
        
        if successful_stocks >= 100:  # 限制最多100檔股票，避免內存問題
            break
    
    print(f"總共成功處理 {successful_stocks} 檔股票")
    
    if not all_stock_combined_data: 
        logger.warning("沒有足夠的股票數據來訓練個股模型。")
        return
    
    # 合併所有股票數據
    stock_train_df = pd.concat(all_stock_combined_data)
    X_stock = stock_train_df[stock_model_feature_cols]
    y_stock = stock_train_df['label']
    
    # 編碼標籤
    stock_label_encoder = LabelEncoder()
    y_stock_encoded = stock_label_encoder.fit_transform(y_stock)
    
    logger.info(f"股票模型原始標籤映射: {list(stock_label_encoder.classes_)} -> {list(range(len(stock_label_encoder.classes_)))}")
    logger.info(f"股票模型訓練數據量: {len(X_stock)} 筆")
    logger.info(f"股票模型標籤分佈: \n{y_stock.value_counts()}")
    
    # 訓練股票模型
    if not X_stock.empty and len(np.unique(y_stock_encoded)) > 1:
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_stock, y_stock_encoded, test_size=TEST_SIZE_RATIO, 
            random_state=RANDOM_STATE, stratify=y_stock_encoded
        )
        
        logger.info("開始訓練個股多空預測模型...")
        
        # 優化的模型參數
        stock_model = xgb.XGBClassifier(
            objective='multi:softmax', 
            num_class=len(stock_label_encoder.classes_), 
            eval_metric='mlogloss', 
            use_label_encoder=False, 
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
        
        # 訓練模型
        stock_model.fit(X_train_s, y_train_s, eval_set=[(X_test_s, y_test_s)], verbose=False)
        
        # 評估模型
        y_pred_s = stock_model.predict(X_test_s)
        logger.info(f"股票模型測試集準確度: {accuracy_score(y_test_s, y_pred_s):.4f}")
        logger.info(f"股票模型分類報告:\n{classification_report(y_test_s, y_pred_s, target_names=[str(c) for c in stock_label_encoder.classes_])}")
        
        # 保存模型
        joblib.dump({'model': stock_model, 'label_encoder': stock_label_encoder}, STOCK_MODEL_SAVE_PATH)
        logger.info(f"個股多空預測模型已儲存至: {STOCK_MODEL_SAVE_PATH}")
    else: 
        logger.warning("沒有足夠或多樣的股票數據來訓練個股模型。")
    
    # --- 訓練市場模型 ---
    logger.info("開始訓練市場情勢判斷模型...")
    
    market_raw_data = prep_data(MARKET_INDEX_ID)
    if market_raw_data is None: 
        logger.error("無法加載大盤數據，無法訓練市場情勢模型。")
        return
    
    market_features = calculate_features(market_raw_data, is_market=True)
    market_labels = create_market_labels(market_raw_data, FUTURE_DAYS_FOR_MARKET_LABEL, MARKET_BULL_THRESHOLD)
    
    combined_market_df = market_features.merge(market_labels.rename('label'), left_index=True, right_index=True)
    combined_market_df = combined_market_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    
    market_model_feature_cols = ['price_change_1d','price_change_5d','ma_20','ma20_slope','ma_60','rsi_14','macd','kd_k','atr_14','bollinger_width']
    combined_market_df = combined_market_df.dropna(subset=market_model_feature_cols + ['label'])
    
    X_market = combined_market_df[market_model_feature_cols]
    y_market = combined_market_df['label']
    
    logger.info(f"市場情勢模型訓練數據量: {len(X_market)} 筆")
    logger.info(f"市場情勢模型標籤分佈: \n{y_market.value_counts()}")
    
    # 訓練市場模型
    if not X_market.empty and len(np.unique(y_market)) > 1:
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
            X_market, y_market, test_size=TEST_SIZE_RATIO, 
            random_state=RANDOM_STATE, stratify=y_market
        )
        
        logger.info("開始訓練市場情勢判斷模型...")
        
        # 優化的市場模型參數
        regime_model = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss', 
            use_label_encoder=False, 
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
        
        # 訓練模型
        regime_model.fit(X_train_m, y_train_m, eval_set=[(X_test_m, y_test_m)], verbose=False)
        
        # 評估模型
        y_pred_m = regime_model.predict(X_test_m)
        logger.info(f"市場情勢模型測試集準確度: {accuracy_score(y_test_m, y_pred_m):.4f}")
        logger.info(f"市場情勢模型分類報告:\n{classification_report(y_test_m, y_pred_m, target_names=['bear', 'bull'])}")
        
        # 保存模型
        joblib.dump(regime_model, REGIME_MODEL_SAVE_PATH)
        logger.info(f"市場情勢判斷模型已儲存至: {REGIME_MODEL_SAVE_PATH}")
    else: 
        logger.warning("沒有足夠或多樣的大盤數據來訓練市場情勢模型。")
    
    logger.info("===== XGBoost 模型訓練完成 (修正版) =====")

if __name__ == '__main__':
    train_models() 