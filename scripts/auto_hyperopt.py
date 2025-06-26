# =====================================================================
# 專案: 機器學習策略專案
# 階段: 自動化超參數調優 - XGBoost 多空雙向模型與市場情勢模型
#
# 功能:
#   - 使用Optuna自動搜尋最佳超參數
#   - 自動訓練多個模型版本並比較
#   - 保存最佳模型與調優報告
#   - 支援股票模型與市場模型分別調優
# =====================================================================

import os
import joblib
import time
import logging
import numpy as np
import pandas as pd
import talib
import xgboost as xgb
import optuna
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import tqdm
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score

# --- 全域設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='hyperopt_training.log', filemode='a')
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

# --- 超參數調優設定 ---
N_TRIALS = 50  # 調優試驗次數
CV_FOLDS = 5   # 交叉驗證折數

# --- 核心函數 (與原腳本相同) ---
def prep_data(stock_id: str) -> Optional[pd.DataFrame]:
    """準備股票數據，避免數據洩漏"""
    file_path = os.path.join(DATA_DIR, f"{stock_id}_history.parquet")
    if not os.path.exists(file_path): 
        return None
    
    try:
        df = pd.read_parquet(file_path)
        df.columns = [c.lower() for c in df.columns]
        
        if 'date' not in df.columns: 
            df = df.reset_index()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').set_index('date')
        
        required = ['high', 'low', 'close', 'open', 'volume']
        
        if any(c not in df.columns for c in required): 
            return None
        
        if df[required].isnull().any().any(): 
            return None
        
        if (df['volume'] <= 0).any(): 
            return None
        
        df['next_day_open'] = df['open'].shift(-1)
        
        return df.dropna(subset=required)
    except Exception as e:
        logger.debug(f"讀取 {stock_id} 數據時出錯: {e}")
        return None

def calculate_features(df: pd.DataFrame, is_market: bool = False) -> pd.DataFrame:
    """計算技術分析特徵"""
    df_feat = df.copy()
    
    # 基本價格變化
    df_feat['price_change_1d'] = df_feat['close'].pct_change(1)
    df_feat['price_change_5d'] = df_feat['close'].pct_change(5)
    df_feat['volume_change_1d'] = df_feat['volume'].pct_change(1)
    
    # 確保數據類型正確
    close_array = df_feat['close'].astype(float).to_numpy()
    high_array = df_feat['high'].astype(float).to_numpy()
    low_array = df_feat['low'].astype(float).to_numpy()
    
    # 移動平均線
    df_feat['ma_5'] = talib.SMA(close_array, 5)
    df_feat['ma_20'] = talib.SMA(close_array, 20)
    df_feat['ma_60'] = talib.SMA(close_array, 60)
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
    
    return df_feat.replace([np.inf, -np.inf], np.nan)

def create_stock_labels(df: pd.DataFrame, future_days: int, buy_threshold: float, sell_threshold: float) -> pd.Series:
    """創建股票交易標籤"""
    if len(df) <= future_days: 
        return pd.Series(dtype=int)
    
    df['future_open'] = df['open'].shift(-future_days)
    df['future_return'] = (df['future_open'] - df['next_day_open']) / df['next_day_open']
    
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[df['future_return'] >= buy_threshold] = 1
    labels[df['future_return'] <= sell_threshold] = -1
    
    return labels.dropna()

def create_market_labels(df: pd.DataFrame, future_days: int, bull_threshold: float) -> pd.Series:
    """創建市場情勢標籤"""
    if len(df) <= future_days: 
        return pd.Series(dtype=int)
    
    df['future_close'] = df['close'].shift(-future_days)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[df['future_return'] >= bull_threshold] = 1
    
    return labels.dropna()

# --- 數據準備函數 ---
def prepare_stock_data() -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """準備股票模型訓練數據"""
    logger.info("準備股票模型訓練數據...")
    
    with open(LIST_PATH, "r", encoding="utf-8") as f:
        stock_list = sorted([l.strip() for l in f if l.strip().isdigit()])
    
    all_stock_combined_data = []
    stock_model_feature_cols = ['price_change_1d','price_change_5d','volume_change_1d','ma_5','ma_20','ma20_slope','ma_60','rsi_14','macd','macdsignal','macdhist','kd_k','kd_d','atr_14','bollinger_width','price_vs_ma20','williams_r']
    
    successful_stocks = 0
    for sid in tqdm.tqdm(stock_list, desc="處理個股數據"):
        df_raw = prep_data(sid)
        if df_raw is None or len(df_raw) < 100:
            continue
            
        df_feat = calculate_features(df_raw, is_market=False)
        labels = create_stock_labels(df_raw, FUTURE_DAYS_FOR_STOCK_LABEL, STOCK_BUY_THRESHOLD, STOCK_SELL_THRESHOLD)
        
        if labels.empty:
            continue
        
        combined_df = df_feat.merge(labels.rename('label'), left_index=True, right_index=True)
        combined_df = combined_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
        valid_df = combined_df.dropna(subset=stock_model_feature_cols + ['label'])
        
        if len(valid_df) >= 50:
            all_stock_combined_data.append(valid_df)
            successful_stocks += 1
        
        if successful_stocks >= 100:
            break
    
    if not all_stock_combined_data:
        raise ValueError("沒有足夠的股票數據")
    
    stock_train_df = pd.concat(all_stock_combined_data)
    X_stock = stock_train_df[stock_model_feature_cols]
    y_stock = stock_train_df['label']
    
    stock_label_encoder = LabelEncoder()
    y_stock_encoded = stock_label_encoder.fit_transform(y_stock)
    
    logger.info(f"股票模型數據準備完成: {len(X_stock)} 筆")
    return X_stock, y_stock_encoded, stock_label_encoder

def prepare_market_data() -> Tuple[pd.DataFrame, pd.Series]:
    """準備市場模型訓練數據"""
    logger.info("準備市場模型訓練數據...")
    
    market_raw_data = prep_data(MARKET_INDEX_ID)
    if market_raw_data is None:
        raise ValueError("無法加載大盤數據")
    
    market_features = calculate_features(market_raw_data, is_market=True)
    market_labels = create_market_labels(market_raw_data, FUTURE_DAYS_FOR_MARKET_LABEL, MARKET_BULL_THRESHOLD)
    
    combined_market_df = market_features.merge(market_labels.rename('label'), left_index=True, right_index=True)
    combined_market_df = combined_market_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    
    market_model_feature_cols = ['price_change_1d','price_change_5d','ma_20','ma20_slope','ma_60','rsi_14','macd','kd_k','atr_14','bollinger_width']
    combined_market_df = combined_market_df.dropna(subset=market_model_feature_cols + ['label'])
    
    X_market = combined_market_df[market_model_feature_cols]
    y_market = combined_market_df['label']
    
    logger.info(f"市場模型數據準備完成: {len(X_market)} 筆")
    return X_market, y_market

# --- 超參數調優目標函數 ---
def stock_objective(trial, X, y):
    """股票模型超參數調優目標函數"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        eval_metric='mlogloss',
        use_label_encoder=False,
        **params
    )
    
    # 使用交叉驗證評估
    scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='f1_weighted')
    return scores.mean()

def market_objective(trial, X, y):
    """市場模型超參數調優目標函數"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        **params
    )
    
    # 使用交叉驗證評估
    scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='f1_weighted')
    return scores.mean()

# --- 主要調優函數 ---
def optimize_stock_model():
    """優化股票模型超參數"""
    logger.info("===== 開始股票模型超參數調優 =====")
    
    # 準備數據
    X_stock, y_stock, stock_label_encoder = prepare_stock_data()
    
    # 創建Optuna研究
    study = optuna.create_study(direction='maximize')
    
    # 開始調優
    study.optimize(lambda trial: stock_objective(trial, X_stock, y_stock), n_trials=N_TRIALS)
    
    logger.info(f"股票模型調優完成，最佳分數: {study.best_value:.4f}")
    logger.info(f"最佳參數: {study.best_params}")
    
    # 使用最佳參數訓練最終模型
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y_stock)),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    })
    
    final_model = xgb.XGBClassifier(**best_params)
    
    # 分割數據並訓練
    X_train, X_test, y_train, y_test = train_test_split(
        X_stock, y_stock, test_size=TEST_SIZE_RATIO, 
        random_state=RANDOM_STATE, stratify=y_stock
    )
    
    final_model.fit(X_train, y_train)
    
    # 評估最終模型
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"最終股票模型 - 準確度: {accuracy:.4f}, F1分數: {f1:.4f}")
    
    # 保存模型和調優結果
    joblib.dump({
        'model': final_model, 
        'label_encoder': stock_label_encoder,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'optimization_history': study.trials_dataframe()
    }, STOCK_MODEL_SAVE_PATH)
    
    # 保存調優報告
    report = {
        'model_type': 'stock_model',
        'best_params': study.best_params,
        'best_score': study.best_value,
        'final_accuracy': accuracy,
        'final_f1': f1,
        'n_trials': N_TRIALS,
        'optimization_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(BASE_PATH, 'stock_model_optimization_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"股票模型已保存至: {STOCK_MODEL_SAVE_PATH}")
    logger.info(f"調優報告已保存至: {os.path.join(BASE_PATH, 'stock_model_optimization_report.json')}")
    
    return final_model, stock_label_encoder, study

def optimize_market_model():
    """優化市場模型超參數"""
    logger.info("===== 開始市場模型超參數調優 =====")
    
    # 準備數據
    X_market, y_market = prepare_market_data()
    
    # 創建Optuna研究
    study = optuna.create_study(direction='maximize')
    
    # 開始調優
    study.optimize(lambda trial: market_objective(trial, X_market, y_market), n_trials=N_TRIALS)
    
    logger.info(f"市場模型調優完成，最佳分數: {study.best_value:.4f}")
    logger.info(f"最佳參數: {study.best_params}")
    
    # 使用最佳參數訓練最終模型
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    })
    
    final_model = xgb.XGBClassifier(**best_params)
    
    # 分割數據並訓練
    X_train, X_test, y_train, y_test = train_test_split(
        X_market, y_market, test_size=TEST_SIZE_RATIO, 
        random_state=RANDOM_STATE, stratify=y_market
    )
    
    final_model.fit(X_train, y_train)
    
    # 評估最終模型
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"最終市場模型 - 準確度: {accuracy:.4f}, F1分數: {f1:.4f}")
    
    # 保存模型和調優結果
    joblib.dump({
        'model': final_model,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'optimization_history': study.trials_dataframe()
    }, REGIME_MODEL_SAVE_PATH)
    
    # 保存調優報告
    report = {
        'model_type': 'market_model',
        'best_params': study.best_params,
        'best_score': study.best_value,
        'final_accuracy': accuracy,
        'final_f1': f1,
        'n_trials': N_TRIALS,
        'optimization_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(BASE_PATH, 'market_model_optimization_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"市場模型已保存至: {REGIME_MODEL_SAVE_PATH}")
    logger.info(f"調優報告已保存至: {os.path.join(BASE_PATH, 'market_model_optimization_report.json')}")
    
    return final_model, study

def main():
    """主函數"""
    logger.info("===== 自動化超參數調優開始 =====")
    start_time = time.time()
    
    try:
        # 調優股票模型
        stock_model, stock_encoder, stock_study = optimize_stock_model()
        
        # 調優市場模型
        market_model, market_study = optimize_market_model()
        
        # 生成總體報告
        total_time = time.time() - start_time
        summary_report = {
            'optimization_summary': {
                'total_time_minutes': total_time / 60,
                'stock_model_best_score': stock_study.best_value,
                'market_model_best_score': market_study.best_value,
                'n_trials_per_model': N_TRIALS,
                'completion_date': datetime.now().isoformat()
            },
            'recommendations': [
                "建議定期重新調優（如每月或每季）",
                "可以嘗試調整N_TRIALS來平衡時間與效果",
                "考慮使用更長的歷史數據來提升模型穩健性"
            ]
        }
        
        with open(os.path.join(BASE_PATH, 'optimization_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"===== 自動化超參數調優完成，總耗時: {total_time/60:.2f} 分鐘 =====")
        logger.info(f"總體報告已保存至: {os.path.join(BASE_PATH, 'optimization_summary.json')}")
        
    except Exception as e:
        logger.error(f"調優過程中發生錯誤: {e}")
        raise

if __name__ == '__main__':
    main() 