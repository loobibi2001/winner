# 檔名: V45_Single_Period_ML_Backtest.py
# =====================================================================
# 專案: 機器學習策略專案
# 階段: 單一期間機器學習策略回測與XGBoost優化
#
# 目標:
#   - 在一個固定的訓練期間內，對XGBoost模型進行超參數優化。
#   - 使用最佳模型在一個固定的測試期間內進行回測。
#   - 不涉及Walk-Forward Optimization的滾動窗口和權益曲線拼接。
#
# 重要更新:
#   - 整合 XGBoost 超參數優化 (GridSearchCV)。
#   - 優化了空頭交易的現金流計算。
#   - 提供單一執行週期，需設定訓練和測試日期。
# =====================================================================

import os
import joblib
import time
import logging
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict

# 導入機器學習相關庫
from sklearn.model_selection import GridSearchCV # 或者 RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer

import matplotlib
matplotlib.use('Agg') # 在非互動模式下運行，例如在伺服器上
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 全域設定 ---
# 配置日誌，同時輸出到文件和控制台
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='run_history_single_period.log', filemode='a')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger = logging.getLogger('')
if not logger.handlers: # 避免重複添加 handler
    logger.addHandler(console_handler)

# --- 數據路徑設定 (請務必修改為您的實際路徑，且路徑中不要有中文或特殊字元) ---
BASE_DIR = r"D:\your_project_folder\winner" # <<<<<<<< 請在這裡修改您的項目根目錄路徑 >>>>>>>>>>
CONFIG = {
    "data_dir": os.path.join(BASE_DIR, "StockData_Parquet"),
    "list_path": os.path.join(BASE_DIR, "stock_list.txt"),
    "regime_model_path": os.path.join(BASE_DIR, "regime_model.joblib"), # 確保此文件存在
    "market_index_id": "TAIEX", # 市場指數ID，請確保數據文件中存在
    "runs_dir": os.path.join(BASE_DIR, "runs") # 輸出結果目錄
}

# --- 策略與風控參數 ---
PREDICTION_PROB_THRESHOLD = 0.55 # 模型預測置信度閾值
ATR_MULTIPLIER = 2.0             # ATR停損/停利倍數
MAX_OPEN_POSITIONS = 10          # 最大同時持倉數量
MAX_LOSS_PER_TRADE_PCT = 0.50    # 單筆交易最大虧損百分比 (用於硬性停損)

# --- 固定時間區間設定 (取代WFO的滾動窗口) ---
TRAIN_START_DATE = "2010-01-01" # 訓練數據起始日期
TRAIN_END_DATE = "2014-12-31"   # 訓練數據結束日期 (例如：5年訓練期)
TEST_START_DATE = "2015-01-01"  # 測試數據起始日期
TEST_END_DATE = "2015-12-31"    # 測試數據結束日期 (例如：1年測試期)

# --- 常量 ---
TRANSACTION_FEE_RATE = 0.001425 # 買賣雙向手續費率 (例如：0.1425%)
TRANSACTION_TAX_RATE = 0.003    # 交易稅率 (僅賣出時徵收，例如：0.3%)
INITIAL_CAPITAL = 10_000_000    # 初始資金
ANNUAL_TRADING_DAYS = 252       # 年交易日數 (用於年化計算)

@dataclass
class BacktestResult:
    """單次回測結果的數據類"""
    params: Dict[str, Any]      # 本次回測使用的參數，包括最佳XGBoost參數
    kpis: Dict[str, Any] = field(default_factory=dict) # 關鍵績效指標
    equity_curve: Optional[pd.Series] = None           # 權益曲線
    trades: Optional[pd.DataFrame] = None              # 交易日誌

# --- 輔助函數 ---

def prep_data(stock_id: str, data_dir: str) -> Optional[pd.DataFrame]:
    """
    載入並初步清理單一股票的歷史數據。
    處理數據缺失值、日期排序，並計算次日開盤價。
    """
    file_path = os.path.join(data_dir, f"{stock_id}_history.parquet")
    if not os.path.exists(file_path):
        logger.debug(f"檔案不存在: {file_path}")
        return None
    try:
        df = pd.read_parquet(file_path)
        df.columns = [c.lower() for c in df.columns]
        if 'date' not in df.columns: df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').set_index('date')
        df = df[df['close'] >= 1] # 過濾掉股價過低的數據
        required = ['high', 'low', 'close', 'open', 'volume']
        if df.empty or any(c not in df.columns for c in required) or df[required].isnull().any().any() or df['volume'].le(0).any():
            logger.debug(f"{stock_id} 數據不完整或包含無效值。")
            return None
        df['next_day_open'] = df['open'].shift(-1)
        return df.dropna(subset=required + ['next_day_open'])
    except Exception as e:
        logger.warning(f"載入或處理 {stock_id} 數據時出錯: {e}")
        return None

def calculate_features(df: pd.DataFrame, is_market: bool = False) -> pd.DataFrame:
    """
    根據策略需求計算所有必要的技術指標和特徵。
    包含多種常用的價量指標。
    """
    df_feat = df.copy()
    # 定義市場和個股的特徵列表
    market_feature_list = ['price_change_1d','price_change_5d','ma_20','ma20_slope','ma_60','rsi_14','macd','kd_k','atr_14','bollinger_width']
    stock_feature_list = market_feature_list + ['volume_change_1d', 'price_vs_ma20', 'williams_r', 'ma_5', 'macdsignal', 'macdhist', 'kd_d']

    # 計算基本價量變化
    df_feat['price_change_1d'] = df_feat['close'].pct_change(1)
    df_feat['price_change_5d'] = df_feat['close'].pct_change(5)
    df_feat['volume_change_1d'] = df_feat['volume'].pct_change(1)

    # 計算移動平均線
    df_feat['ma_5'] = talib.SMA(df_feat['close'], 5)
    df_feat['ma_20'] = talib.SMA(df_feat['close'], 20)
    
    # 加入ma20_slope特徵計算
    df_feat['ma20_slope'] = df_feat['ma_20'].pct_change(1)
    
    df_feat['ma_60'] = talib.SMA(df_feat['close'], 60)

    # 計算RSI
    df_feat['rsi_14'] = talib.RSI(df_feat['close'], 14)

    # 計算MACD
    df_feat['macd'], df_feat['macdsignal'], df_feat['macdhist'] = talib.MACD(df_feat['close'], 12, 26, 9)

    # 計算KD (Stochastic Oscillator)
    df_feat['kd_k'], df_feat['kd_d'] = talib.STOCH(df_feat['high'], df_feat['low'], df_feat['close'], 9, 3, 3)

    # 計算ATR
    df_feat['atr_14'] = talib.ATR(df_feat['high'], df_feat['low'], df_feat['close'], 14)

    # 計算布林帶寬度
    upper, middle, lower = talib.BBANDS(df_feat['close'], 20)
    df_feat['bollinger_width'] = np.where(middle > 0, (upper - lower) / middle, 0)

    # 僅為個股計算的特徵
    if not is_market:
        df_feat['price_vs_ma20'] = np.where(df_feat['ma_20'] > 0, df_feat['close'] / df_feat['ma_20'], 1)
        df_feat['williams_r'] = talib.WILLR(df_feat['high'], df_feat['low'], df_feat['close'], 14)

    # 根據是否為市場指數選擇最終的特徵列
    final_features = market_feature_list if is_market else stock_feature_list
    # 清理無限值和NaN，確保模型訓練數據的品質
    return df_feat.replace([np.inf, -np.inf], np.nan).dropna(subset=final_features)

def create_long_short_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    為機器學習模型創建特徵和多空標籤。
    標籤基於未來20天的報酬率：
    1: 未來報酬率 > 5% (做多信號)
    -1: 未來報酬率 < -5% (做空信號)
    0: 其他 (中性/不交易)
    """
    df_with_features = calculate_features(df, is_market=False)
    if 'close' not in df_with_features.columns: # 確保有收盤價用於計算未來報酬
        df_with_features = df_with_features.join(df['close'])

    df_with_features['future_return'] = (df_with_features['close'].shift(-20) - df_with_features['close']) / df_with_features['close']
    conditions = [df_with_features['future_return'] > 0.05, df_with_features['future_return'] < -0.05]
    choices = [1, -1]
    df_with_features['label'] = np.select(conditions, choices, default=0)
    return df_with_features.dropna(subset=['future_return', 'label'])

def run_single_period_backtest(train_start, train_end, test_start, test_end, stock_list, config) -> BacktestResult:
    """
    執行單一訓練、超參數優化和回測週期。
    此函數包含了模型訓練、XGBoost超參數優化、市場機制判斷以及多空交易回測的核心邏輯。
    """
    logger.info(f"開始執行單一週期回測: 訓練 {train_start} to {train_end}, 測試 {test_start} to {test_end}")

    # 1. 數據準備與模型訓練
    # 加載所有股票數據
    all_stock_data = {sid: df for sid in stock_list if (df := prep_data(sid, config['data_dir'])) is not None}

    # 聚合訓練數據集 (從所有股票中提取訓練期間的數據)
    train_features_list = [
        create_long_short_features_and_labels(df.loc[train_start:train_end])
        for df in all_stock_data.values()
        if not df.loc[train_start:train_end].empty # 確保訓練數據不為空
    ]
    if not train_features_list:
        logger.warning(f"訓練數據為空，無法執行回測。訓練區間: {train_start} to {train_end}")
        return BacktestResult(params={'train': f"{train_start}-{train_end}", 'test': f"{test_start}-{test_end}"}, kpis={'錯誤': '訓練數據為空'})

    train_dataset = pd.concat(train_features_list, ignore_index=True)

    X_train = train_dataset.drop(columns=['label', 'future_return'], errors='ignore')
    y_train = train_dataset['label']

    # 使用LabelEncoder轉換標籤，確保XGBoost能處理多分類
    le = LabelEncoder()
    # 預先定義 classes，確保所有可能的標籤 (1, -1, 0) 都被正確編碼，即使在當前訓練數據中某些標籤缺失
    le.fit(np.array([1, -1, 0]))
    y_train_encoded = le.transform(y_train)

    # --- XGBoost 超參數網格搜索 (GridSearchCV) ---
    # 定義 XGBoost 參數網格 (您可以根據您的計算能力和調優需求調整這些參數和範圍)
    param_grid = {
        'n_estimators': [50, 100],        # 決策樹的數量
        'learning_rate': [0.05, 0.1],   # 學習率
        'max_depth': [3, 5],             # 樹的最大深度
        'subsample': [0.8, 1.0],       # 訓練樣本的子採樣比例
        'colsample_bytree': [0.8, 1.0],# 每棵樹的列（特徵）子採樣比例
        # 更多參數，如 'gamma': [0, 0.1], 'lambda': [1, 10], 'alpha': [0, 0.1]
    }

    # 定義評分器：F1-score 的 'macro' 平均對於多分類且可能不平衡的標籤是個好選擇
    scorer = make_scorer(f1_score, average='macro')

    # 初始化 XGBoost 分類器基礎模型
    xgb_base_model = XGBClassifier(objective='multi:softprob', eval_metric="mlogloss",
                                   use_label_encoder=False, n_jobs=-1, random_state=42)

    # 執行網格搜索
    logger.info(f"開始在訓練區間 {train_start}-{train_end} 進行 XGBoost 超參數優化...")
    grid_search = GridSearchCV(estimator=xgb_base_model,
                               param_grid=param_grid,
                               scoring=scorer,
                               cv=3,       # 交叉驗證摺疊數，可調整為5或其他
                               verbose=1,  # 設置為1或2可以查看更多訓練進度
                               n_jobs=-1)  # 使用所有可用的CPU核心進行并行計算

    grid_search.fit(X_train, y_train_encoded)

    # 獲取最佳模型和最佳參數
    stock_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"訓練區間 {train_start}-{train_end}: 最佳 XGBoost 參數: {best_params}")
    logger.info(f"訓練區間 {train_start}-{train_end}: 訓練集最佳 F1-score (Macro): {best_score:.4f}")

    # 載入市場機制模型 (假設已提前訓練並保存)
    try:
        regime_model = joblib.load(config['regime_model_path'])
    except FileNotFoundError:
        logger.error(f"市場機制模型檔案未找到: {config['regime_model_path']}")
        return BacktestResult(params={'train': f"{train_start}-{train_end}", 'test': f"{test_start}-{test_end}"}, kpis={'錯誤': '市場機制模型檔案未找到'})


    # 計算測試集所需的市場和股票特徵
    market_raw_data = prep_data(config['market_index_id'], config['data_dir'])
    if market_raw_data is None:
        logger.error(f"無法載入市場指數數據: {config['market_index_id']}")
        return BacktestResult(params={'train': f"{train_start}-{train_end}", 'test': f"{test_start}-{test_end}"}, kpis={'錯誤': '無法載入市場指數數據'})
    market_features = calculate_features(market_raw_data, is_market=True)
    stock_features = {sid: calculate_features(df) for sid, df in all_stock_data.items()}

    # 確保特徵列與模型訓練時一致
    stock_feature_cols = stock_model.feature_names_in_
    market_feature_cols = regime_model.feature_names_in_

    # 2. 回測核心邏輯 (在指定的測試區間執行)
    cash = INITIAL_CAPITAL # 回測從初始資金開始
    positions = {}         # 當前持有的倉位
    trades_log = []        # 記錄所有完成的交易
    test_date_range = pd.date_range(start=test_start, end=test_end, freq='B') # 根據商業日曆生成測試日期範圍
    equity_curve = pd.Series(index=test_date_range, dtype=float) # 記錄每日權益

    logger.info(f"開始在測試區間 {test_start}-{test_end} 執行回測...")
    for current_date in test_date_range:
        if current_date > datetime.now(): # 避免回測到未來日期
            break

        # --- A. 市場機制判斷 (每日更新) ---
        is_bull_market = True # 預設為牛市
        if current_date in market_features.index:
            features = market_features.loc[[current_date]][market_feature_cols]
            if not features.empty:
                # 假設市場機制模型輸出 1 為牛市，0 為熊市
                is_bull_market = (regime_model.predict(features)[0] == 1)

        # --- B. 管理現有倉位 (檢查出場條件並執行平倉) ---
        # 遍歷當前所有持倉的股票，檢查是否滿足出場條件
        for stock_id in list(positions.keys()):
            pos = positions.get(stock_id)
            # 確保有該股票當日數據且部位存在
            if not pos or stock_id not in all_stock_data or current_date not in all_stock_data[stock_id].index:
                continue

            row = all_stock_data[stock_id].loc[current_date] # 獲取當日數據
            exit_price, exit_reason = row['next_day_open'], None # 預設出場價為次日開盤價
            if pd.isna(exit_price) or exit_price <= 0: exit_price = row['close'] # 若次日開盤價不可用，則用當日收盤價

            # 1. 檢查最大虧損上限 (硬性停損)
            if pos['direction'] == 'long': # 多頭部位
                # 計算從入場價到當前出場價的百分比虧損
                if (pos['entry_price'] - exit_price) / pos['entry_price'] > MAX_LOSS_PER_TRADE_PCT:
                    exit_reason = 'MaxLoss_Capped'
                    # 強制出場價為最大虧損價
                    exit_price = pos['entry_price'] * (1 - MAX_LOSS_PER_TRADE_PCT)
            elif pos['direction'] == 'short': # 空頭部位
                # 計算從入場價到當前出場價的百分比虧損 (空頭是價格上漲虧損)
                if (exit_price - pos['entry_price']) / pos['entry_price'] > MAX_LOSS_PER_TRADE_PCT:
                    exit_reason = 'MaxLoss_Capped'
                    # 強制出場價為最大虧損價
                    exit_price = pos['entry_price'] * (1 + MAX_LOSS_PER_TRADE_PCT)

            # 2. 檢查市場機制轉換出場 (軟性停損/策略性出場)
            if not exit_reason: # 如果尚未觸發最大虧損
                if (pos['direction'] == 'long' and not is_bull_market) or \
                   (pos['direction'] == 'short' and is_bull_market):
                    exit_reason = 'Regime_Exit' # 市場機制不再支持當前方向

            # 3. 檢查停損 (初始停損或ATR移動停損)
            if not exit_reason: # 如果尚未觸發上述任何出場條件
                if pos['direction'] == 'long' and row['low'] <= pos['stop_loss']:
                    exit_reason = 'StopLoss'
                    # 實際出場價取當日最低價和停損價的較差者 (對於多頭是較高者，減少滑價影響)
                    exit_price = max(row['low'], pos['stop_loss'])
                elif pos['direction'] == 'short' and row['high'] >= pos['stop_loss']:
                    exit_reason = 'StopLoss'
                    # 實際出場價取當日最高價和停損價的較差者 (對於空頭是較低者，減少滑價影響)
                    exit_price = min(row['high'], pos['stop_loss'])

            # 如果觸發了任何出場條件，則執行平倉
            if exit_reason:
                pnl = 0
                if pos['direction'] == 'long':
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    # 多頭平倉：現金增加賣出所得，同時扣除賣出手續費和交易稅
                    cash += exit_price * pos['shares'] * (1 - TRANSACTION_FEE_RATE - TRANSACTION_TAX_RATE)
                    logger.info(f"{current_date.date()}: 平倉 {stock_id} (多頭) @ {exit_price:.2f}, 損益: {pnl:,.0f} ({exit_reason})")
                else: # pos['direction'] == 'short' (空頭部位)
                    pnl = (pos['entry_price'] - exit_price) * pos['shares'] # 空頭盈虧計算 (賣出價 - 買入價)
                    # 空頭平倉是買入回補動作，只扣除買入時的手續費。
                    # 交易稅在開倉賣出時已處理。
                    cash -= exit_price * pos['shares'] * (1 + TRANSACTION_FEE_RATE)
                    logger.info(f"{current_date.date()}: 平倉 {stock_id} (空頭) @ {exit_price:.2f}, 損益: {pnl:,.0f} ({exit_reason})")

                # 記錄這筆交易到日誌
                trades_log.append({
                    'stock_id': stock_id,
                    'pnl_dollar_gross': pnl,
                    'direction': pos['direction'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'entry_date_dt': pos['entry_date'],
                    'trade_date_dt': current_date, # 實際平倉日期
                    'exit_reason': exit_reason,
                    'shares': pos['shares']
                })
                del positions[stock_id] # 從持倉列表中移除已平倉的部位

        # --- C. 尋找新的交易機會 (滿足進場條件並開倉) ---
        if len(positions) < MAX_OPEN_POSITIONS: # 檢查是否還有可開倉的空間
            portfolio_value = cash + sum(p.get('value', 0) for p in positions.values()) # 計算當前總權益

            # 遍歷所有股票，尋找交易機會
            for stock_id, df_feat in stock_features.items():
                # 跳過已滿倉、已持倉或無當日數據的股票
                if len(positions) >= MAX_OPEN_POSITIONS or stock_id in positions or current_date not in df_feat.index:
                    continue

                features_today = df_feat.loc[[current_date]][stock_feature_cols]
                if features_today.empty:
                    continue

                # 獲取模型對當日特徵的預測概率和最可信的標籤
                pred_probs = stock_model.predict_proba(features_today)[0]
                pred_label_encoded = np.argmax(pred_probs)
                pred_label = le.inverse_transform([pred_label_encoded])[0] # 轉換回原始標籤 (1, -1, 0)
                pred_confidence = pred_probs[pred_label_encoded]

                # 如果模型預測為中性 (0) 或者置信度不足，則不考慮交易
                if pred_label == 0 or pred_confidence <= PREDICTION_PROB_THRESHOLD:
                    continue

                # 判斷是否符合市場機制下的交易方向 (牛市預測做多，熊市預測做空)
                if not ((is_bull_market and pred_label == 1) or (not is_bull_market and pred_label == -1)):
                    continue # 市場機制與模型預測方向不符，跳過

                entry_price, atr = df_feat.loc[current_date, ['next_day_open', 'atr_14']]
                # 檢查進場價格和ATR的有效性
                if pd.isna(entry_price) or entry_price <= 0 or pd.isna(atr) or atr <= 0:
                    continue

                # 部位大小計算: 基於總權益和每筆交易的風險（假設部署總資產的1.5%）
                capital_to_deploy = portfolio_value * 0.015
                # 將資金轉換為股數，並向下取整到1000股 (台股一張)
                shares = int(capital_to_deploy / entry_price / 1000) * 1000
                
                if shares < 1000: # 確保達到最小交易單位
                    continue

                # 多頭進場邏輯
                if pred_label == 1: # 預測做多
                    buy_cost = shares * entry_price * (1 + TRANSACTION_FEE_RATE) # 計算買入成本 (含手續費)
                    if cash >= buy_cost: # 檢查現金是否足夠
                        cash -= buy_cost # 從現金中扣除買入成本
                        positions[stock_id] = {
                            'shares': shares,
                            'entry_price': entry_price,
                            'entry_date': current_date, # 紀錄進場日期
                            'stop_loss': entry_price - atr * ATR_MULTIPLIER, # 設定多頭停損價
                            'direction': 'long',
                            'value': shares * entry_price # 當前持倉市值 (用於計算總權益)
                        }
                        logger.info(f"{current_date.date()}: 進場 {stock_id} (多頭) @ {entry_price:.2f}, 股數: {shares}")
                # 空頭進場邏輯
                elif pred_label == -1: # 預測做空
                    # 做空時，假設賣出後立即收到現金流，並扣除賣出手續費和交易稅
                    # 這裡的現金增加是模擬賣空所得，用於後續的盈虧結算和資產計算。
                    # 注意：此處簡化處理，未明確模擬保證金帳戶或實際保證金需求。
                    cash_received_from_short = shares * entry_price * (1 - TRANSACTION_FEE_RATE - TRANSACTION_TAX_RATE)
                    cash += cash_received_from_short # 將賣空所得計入總現金
                    positions[stock_id] = {
                        'shares': shares,
                        'entry_price': entry_price,
                        'entry_date': current_date, # 紀錄進場日期
                        'stop_loss': entry_price + atr * ATR_MULTIPLIER, # 設定空頭停損價
                        'direction': 'short',
                        'value': shares * entry_price # 空頭部位的初始價值 (賣出價*股數)
                    }
                    logger.info(f"{current_date.date()}: 進場 {stock_id} (空頭) @ {entry_price:.2f}, 股數: {shares}")


        # --- D. 每日結算權益曲線 ---
        holdings_value = 0
        for stock_id, pos in positions.items():
            # 獲取當前持倉的市價，如果當日數據缺失，則使用入場價格
            current_price = all_stock_data[stock_id].loc[current_date, 'close'] if current_date in all_stock_data[stock_id].index else pos['entry_price']
            
            if pos.get('direction') == 'long':
                pos['value'] = pos['shares'] * current_price # 多頭持倉市值
            else: # Short position (空頭部位)
                # 空頭部位的當前價值計算方式：初始賣空所得 + 浮動盈虧
                # 浮動盈虧 = (賣出價 - 當前價) * 股數
                pos['value'] = (pos['shares'] * pos['entry_price']) + (pos['entry_price'] - current_price) * pos['shares']
            holdings_value += pos.get('value', 0) # 累加所有持倉的價值

        current_equity = cash + holdings_value # 總權益 = 現金 + 持倉總價值
        equity_curve.loc[current_date] = current_equity # 記錄當日總權益

    logger.info(f"單一週期回測結束。")
    return BacktestResult(
        params={'train': f"{train_start}-{train_end}", 'test': f"{test_start}-{test_end}", 'best_xgb_params': best_params}, # 記錄最佳參數
        trades=pd.DataFrame(trades_log),
        equity_curve=equity_curve.dropna() # 返回不含NaN的權益曲線
    )


def calculate_final_kpis(equity_curve: pd.Series, trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    計算關鍵績效指標 (KPIs)。
    包括最終權益、總淨利、年化報酬率、最大回撤、夏普比率、卡瑪比率、總交易次數、勝率和獲利因子。
    """
    kpis = {}
    if not equity_curve.empty:
        final_equity, initial_capital = equity_curve.iloc[-1], equity_curve.iloc[0]
        returns = equity_curve.pct_change().dropna() # 計算每日報酬率
        total_years = len(equity_curve) / ANNUAL_TRADING_DAYS # 計算總回測年數

        kpis['最終權益'] = final_equity
        kpis['總淨利 ($)'] = final_equity - initial_capital
        # 年化報酬率計算
        kpis['年化報酬率 (%)'] = ((final_equity / initial_capital) ** (1 / total_years) - 1) * 100 if total_years > 0 and initial_capital > 0 else 0.0

        # 最大回撤計算
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        kpis['最大回撤 (%)'] = drawdown.min() * 100 if not drawdown.empty else 0.0

        # 夏普比率計算 (假設無風險利率為0)
        kpis['夏普比率'] = (returns.mean() / returns.std()) * np.sqrt(ANNUAL_TRADING_DAYS) if not returns.empty and returns.std() != 0 else 0.0
        
        # 卡瑪比率計算
        if kpis.get('最大回撤 (%)', 0) < 0: # 只有當有回撤時才計算卡瑪比率
            kpis['卡瑪比率'] = kpis['年化報酬率 (%)'] / abs(kpis['最大回撤 (%)'])
        else: # 如果沒有回撤 (最大回撤為0或正)，則根據年化報酬判斷
            kpis['卡瑪比率'] = np.inf if kpis.get('年化報酬率 (%)',0) > 0 else 0.0 # 有收益無回撤為無限大，無收益無回撤為0

    if not trades_df.empty:
        wins = trades_df[trades_df['pnl_dollar_gross'] > 0] # 獲利交易
        losses = trades_df[trades_df['pnl_dollar_gross'] < 0] # 虧損交易 (只考慮實際虧損)
        
        kpis['總交易次數'] = len(trades_df)
        kpis['勝率 (%)'] = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0.0
        
        total_gross_profit = wins['pnl_dollar_gross'].sum() # 總毛利
        total_gross_loss = abs(losses['pnl_dollar_gross'].sum()) # 總毛損 (取絕對值)
        
        # 獲利因子計算
        if total_gross_loss > 0:
            kpis['獲利因子'] = total_gross_profit / total_gross_loss
        else:
            kpis['獲利因子'] = np.inf if total_gross_profit > 0 else 0.0 # 無虧損但有收益為無限大，無虧損無收益為0
    return kpis

def format_kpis_for_display(kpis: Dict[str, Any]) -> str:
    """將KPI字典格式化為易於閱讀的字串，用於日誌和報告輸出。"""
    lines = ["\n" + "="*40, "   << 單一期間機器學習策略績效報告 >>", "="*40]
    lines.append("\n--- 總體績效概覽 ---")
    lines.append(f"{'最終權益':<20}: {kpis.get('最終權益', 0):,.2f}")
    lines.append(f"{'淨利 ($)':<20}: {kpis.get('總淨利 ($)', 0):,.2f}")
    lines.append(f"{'年化報酬率 (%)':<20}: {kpis.get('年化報酬率 (%)', 0):.2f}")
    lines.append(f"{'最大回撤 (%)':<20}: {kpis.get('最大回撤 (%)', 0):.2f}")
    lines.append(f"{'夏普比率':<20}: {kpis.get('夏普比率', 0):.2f}")
    lines.append(f"{'卡瑪比率':<20}: {kpis.get('卡瑪比率', 0):.2f}")
    lines.append("\n--- 交易行為分析 ---")
    lines.append(f"{'總交易次數':<20}: {kpis.get('總交易次數', 0)}")
    lines.append(f"{'勝率 (%)':<20}: {kpis.get('勝率 (%)', 0):.2f}")
    lines.append(f"{'獲利因子':<20}: {kpis.get('獲利因子', 0):.2f}")
    lines.append("\n" + "="*40 + "\n")
    return "\n".join(lines)

def plot_equity_curve(equity_curve, charts_dir, kpis, period_name="回測期間"):
    """繪製權益曲線圖，並保存到指定目錄。"""
    try:
        # 嘗試設定中文字體和解決負號亂碼問題
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        logger.warning(f"警告：無法設定中文字體或負號顯示: {e}")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(equity_curve.index, equity_curve, label='權益曲線', color='blue', linewidth=1.5)
    
    title = f'{period_name} 機器學習策略權益曲線\n'
    kpi_text = (f"年化報酬率: {kpis.get('年化報酬率 (%)', 0):.2f}% | "
                f"最大回撤: {kpis.get('最大回撤 (%)', 0):.2f}% | "
                f"夏普比率: {kpis.get('夏普比率', 0):.2f}")
    
    ax.set_title(title + kpi_text, fontsize=12)
    ax.set_xlabel('日期')
    ax.set_ylabel('總權益 (NTD)')
    ax.grid(True, linestyle='--', alpha=0.6) # 增加網格線
    ax.legend() # 顯示圖例
    
    # 根據測試期日期生成文件名
    start_year = equity_curve.index.min().year
    end_year = equity_curve.index.max().year
    file_name = f"equity_curve_{start_year}_{end_year}.png"

    plt.savefig(os.path.join(charts_dir, file_name), dpi=300, bbox_inches='tight')
    plt.close(fig) # 關閉圖形，釋放記憶體

def main_single_period():
    """
    主執行函數，用於協調單一期間的機器學習策略回測。
    設定回測日期，執行回測，並生成報告和圖表。
    """
    # 創建一個帶時間戳的運行目錄，用於保存本次回測的所有結果
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = os.path.join(CONFIG['runs_dir'], f"run_SinglePeriodML_{run_timestamp}")
    os.makedirs(report_dir, exist_ok=True) # 確保目錄存在

    logger.info(f"===== 單一期間機器學習策略回測啟動: {run_timestamp} =====")
    
    # 讀取股票列表文件
    try:
        with open(CONFIG['list_path'], "r", encoding="utf-8") as f:
            stock_list = sorted([l.strip() for l in f if l.strip().isdigit()])
    except FileNotFoundError:
        logger.error(f"錯誤：找不到股票列表檔案 {CONFIG['list_path']}！請檢查路徑設定。"); return

    logger.info(f"設定訓練期間: {TRAIN_START_DATE} 至 {TRAIN_END_DATE}")
    logger.info(f"設定測試期間: {TEST_START_DATE} 至 {TEST_END_DATE}")

    # 執行單一週期的回測 (包含訓練和優化)
    result = run_single_period_backtest(
        TRAIN_START_DATE, TRAIN_END_DATE,
        TEST_START_DATE, TEST_END_DATE,
        stock_list, CONFIG
    )

    # 檢查回測是否成功執行
    if '錯誤' in result.kpis:
        logger.error(f"回測執行失敗: {result.kpis.get('錯誤')}")
        return

    logger.info("回測完成，開始計算並生成報告...")
    # 計算並顯示最終的關鍵績效指標
    kpis = calculate_final_kpis(result.equity_curve, result.trades)
    kpi_report_str = format_kpis_for_display(kpis)
    logger.info(kpi_report_str)

    # --- 儲存所有結果 ---
    # 儲存KPI報告到文本文件
    with open(os.path.join(report_dir, "single_period_kpi_report.txt"), "w", encoding="utf-8") as f:
        f.write(kpi_report_str)
    
    # 儲存最佳XGBoost參數到文本文件
    if 'best_xgb_params' in result.params:
        with open(os.path.join(report_dir, "best_xgb_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"最佳 XGBoost 參數: {result.params['best_xgb_params']}\n")

    # 儲存交易日誌到CSV
    result.trades.to_csv(os.path.join(report_dir, "single_period_trades_log.csv"), index=False, encoding='utf-8-sig')
    # 儲存權益曲線到CSV
    result.equity_curve.to_csv(os.path.join(report_dir, "single_period_equity_curve.csv"), encoding='utf-8-sig')
    
    # 繪製權益曲線圖並保存
    plot_equity_curve(result.equity_curve, report_dir, kpis, period_name=f"測試期_{TEST_START_DATE.split('-')[0]}-{TEST_END_DATE.split('-')[0]}")
    
    logger.info(f"報告已完整儲存至: {report_dir}")

if __name__ == '__main__':
    main_single_period()