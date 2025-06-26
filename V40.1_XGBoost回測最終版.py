# 檔名: V40.1_XGBoost回測最終版.py
# =====================================================================
# 專案: 機器學習策略專案
# 階段: 最終章 - 整合與回測 XGBoost 多空雙向模型 (完整修正版)
#
# 本次更新:
#   - 補全所有省略的程式碼，解決 silent exit 問題。
#   - 這是一個完整、獨立、可執行的最終腳本。
# =====================================================================

import os
import joblib
import time
import logging
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import tqdm

import matplotlib
matplotlib.use('Agg') # 在非圖形介面環境下運行
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 全域設定 ---
# 配置日誌記錄器，將日誌輸出到文件和控制台
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='run_history.log', filemode='a')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger = logging.getLogger('')
logger.addHandler(console_handler)

BASE_PATH = r"D:\飆股篩選\winner"
WINNER_DIR = BASE_PATH
DATA_DIR = os.path.join(BASE_PATH, "StockData_Parquet")
LIST_PATH = os.path.join(BASE_PATH, "stock_list.txt")
STOCK_MODEL_PATH = os.path.join(BASE_PATH, "xgboost_long_short_model.joblib")
REGIME_MODEL_PATH = os.path.join(BASE_PATH, "regime_model.joblib")
MARKET_INDEX_ID = "TAIEX"

# --- 策略參數 ---
PREDICTION_PROB_THRESHOLD = 0.35 # 模型預測機率門檻，降低以增加交易機會
ATR_MULTIPLIER = 2.0             # ATR止損/止盈乘數
MAX_OPEN_POSITIONS = 15          # 最大同時持倉股票數量，增加以提高交易頻率

# --- 常量 ---
TRANSACTION_FEE_RATE = 0.001425 # 交易手續費率 (萬分之1.425)
TRANSACTION_TAX_RATE = 0.003    # 交易稅率 (千分之3)
INITIAL_CAPITAL = 10_000_000    # 初始資金
ANNUAL_TRADING_DAYS = 252       # 年化交易日數

# === 新增風控參數 ===
MAX_LOSS_PER_TRADE = 0.10  # 單筆最大虧損10%
TAKE_PROFIT = 0.10        # 固定停利10%
USE_RSI_FILTER = True     # 是否啟用RSI輔助
RSI_LONG_THRESHOLD = 35   # RSI低於此值才考慮做多
RSI_SHORT_THRESHOLD = 65  # RSI高於此值才考慮做空

# === 新增持倉與回撤風控參數 ===
MAX_POSITION_PCT = 0.10   # 單一股票最大持倉比例10%
MAX_DRAWDOWN_PCT = 0.30   # 組合最大回撤警戒30%
drawdown_triggered = False

# --- 資料類別 ---
@dataclass
class BacktestResult:
    """
    用於儲存回測結果的資料類別。
    """
    params: Dict[str, Any]
    kpis: Dict[str, Any] = field(default_factory=dict)
    equity_curve: Optional[pd.Series] = None
    trades: Optional[pd.DataFrame] = None

# --- 所有功能函數 ---

def prep_data(stock_id: str) -> Optional[pd.DataFrame]:
    """
    準備個股或大盤的歷史數據。
    讀取parquet文件，清理並標準化數據格式。
    """
    file_path = os.path.join(DATA_DIR, f"{stock_id}_history.parquet")
    if not os.path.exists(file_path):
        logger.debug(f"檔案不存在: {file_path}")
        return None
    try:
        df = pd.read_parquet(file_path)
        df.columns = [c.lower() for c in df.columns]
        if 'date' not in df.columns:
            df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').set_index('date')
        required = ['high', 'low', 'close', 'open', 'volume']
        
        # 修正數據驗證邏輯
        if any(c not in df.columns for c in required):
            logger.debug(f"{stock_id} 缺少必要欄位")
            return None
        
        # 檢查是否有空值
        if df[required].isnull().to_numpy().any():
            logger.debug(f"{stock_id} 數據包含空值")
            return None
        
        # 檢查交易量是否為正
        if (df['volume'] <= 0).any():
            logger.debug(f"{stock_id} 數據包含無效的交易量")
            return None
            
        # 計算次日開盤價，用於模擬實際交易的進出價
        df['next_day_open'] = df['open'].shift(-1)
        # 自動補上漲跌停欄位（如不存在）
        if 'limit_up' not in df.columns:
            df['limit_up'] = df['close'] * 1.1
        if 'limit_down' not in df.columns:
            df['limit_down'] = df['close'] * 0.9
        # 移除包含NaN值的行，特別是最後一天的next_day_open
        return df.dropna(subset=required + ['next_day_open'])
    except Exception as e:
        logger.debug(f"讀取 {stock_id} 數據時出錯: {e}")
        return None

def calculate_features(df: pd.DataFrame, is_market: bool = False) -> pd.DataFrame:
    """
    計算股票或市場指數的技術分析特徵。
    """
    df_feat = df.copy()
    # 定義市場和股票的特徵列表，確保訓練和預測時使用相同的特徵集合
    
    # ===== 步驟 1: 在特徵列表中加入 'ma20_slope' =====
    market_feature_list = ['price_change_1d','price_change_5d','ma_20','ma20_slope','ma_60','rsi_14','macd','kd_k','atr_14','bollinger_width']
    stock_feature_list = market_feature_list + ['volume_change_1d', 'price_vs_ma20', 'williams_r', 'ma_5', 'macdsignal', 'macdhist', 'kd_d']

    df_feat['price_change_1d'] = df_feat['close'].pct_change(1)
    df_feat['price_change_5d'] = df_feat['close'].pct_change(5)
    df_feat['volume_change_1d'] = df_feat['volume'].pct_change(1)
    
    # 修正TA-Lib函數調用
    df_feat['ma_5'] = talib.SMA(df_feat['close'].values, 5)  # type: ignore
    df_feat['ma_20'] = talib.SMA(df_feat['close'].values, 20)  # type: ignore

    # ===== 步驟 2: 加入計算斜率的程式碼 =====
    # 計算20日均線的斜率 (這裡用1日的百分比變化來近似)
    df_feat['ma20_slope'] = df_feat['ma_20'].pct_change(1)

    df_feat['ma_60'] = talib.SMA(df_feat['close'].values, 60)  # type: ignore
    df_feat['rsi_14'] = talib.RSI(df_feat['close'].values, 14)  # type: ignore
    df_feat['macd'], df_feat['macdsignal'], df_feat['macdhist'] = talib.MACD(df_feat['close'].values, 12, 26, 9)  # type: ignore
    df_feat['kd_k'], df_feat['kd_d'] = talib.STOCH(df_feat['high'].values, df_feat['low'].values, df_feat['close'].values, 9, 3, 3)  # type: ignore
    df_feat['atr_14'] = talib.ATR(df_feat['high'].values, df_feat['low'].values, df_feat['close'].values, 14)  # type: ignore
    upper, middle, lower = talib.BBANDS(df_feat['close'].values, 20)  # type: ignore
    df_feat['bollinger_width'] = np.where(middle > 0, (upper - lower) / middle, 0) # 避免除以零

    if not is_market:
        df_feat['price_vs_ma20'] = np.where(df_feat['ma_20'] > 0, df_feat['close'] / df_feat['ma_20'], 1) # 避免除以零
        df_feat['williams_r'] = talib.WILLR(df_feat['high'].values, df_feat['low'].values, df_feat['close'].values, 14)  # type: ignore

    final_features = market_feature_list if is_market else stock_feature_list
    # 移除包含NaN/inf值的行，這些值會影響模型預測
    return df_feat.replace([np.inf, -np.inf], np.nan).dropna(subset=final_features)

def run_xgboost_long_short_backtest(stock_model_data, regime_model, all_stock_data, market_features):
    """
    執行基於XGBoost模型的多空雙向交易策略回測。
    """
    stock_model, stock_label_encoder = stock_model_data['model'], stock_model_data['label_encoder']

    logger.info("為所有個股計算回測所需特徵...")
    # 提前計算所有股票的特徵，提高回測效率
    stock_features = {sid: calculate_features(df) for sid, df in tqdm.tqdm(all_stock_data.items(), desc="計算個股特徵")}

    # 從模型中獲取特徵名稱，確保使用的特徵與訓練時一致
    stock_feature_cols = stock_model.feature_names_in_
    # 處理新格式的市場模型（可能是字典格式）
    if isinstance(regime_model, dict):
        market_model = regime_model['model']
        market_feature_cols = market_model.feature_names_in_
    else:
        market_model = regime_model
        market_feature_cols = regime_model.feature_names_in_

    cash, positions, trades_log = INITIAL_CAPITAL, {}, []
    
    # 修正：動態確定實際的數據日期範圍
    all_stock_dates = []
    for stock_id, df in all_stock_data.items():
        if not df.empty:
            all_stock_dates.extend(df.index.tolist())
    
    if not all_stock_dates:
        logger.error("沒有找到任何股票數據，無法執行回測")
        return None
    
    # 找出所有股票數據的日期範圍
    min_date = min(all_stock_dates)
    max_date = max(all_stock_dates)
    
    # 創建只包含實際數據日期的權益曲線
    all_dates = pd.date_range(start=min_date, end=max_date, freq='B')
    equity_curve = pd.Series(index=all_dates, dtype=float)
    
    # 初始化權益曲線的起始值
    equity_curve.iloc[0] = INITIAL_CAPITAL

    # 添加統計變數
    bear_market_days = 0
    short_attempts = 0
    short_executed = 0

    logger.info("開始執行回測...")
    for current_date in tqdm.tqdm(equity_curve.index, desc="執行回測"):
        # 1. 判斷市場情勢
        is_bull_market = True # 預設為牛市，如果大盤數據不足則保持預設
        if current_date in market_features.index:
            features = market_features.loc[[current_date]][market_feature_cols]
            if not features.empty:
                # 預測市場情勢，將編碼後的結果轉換回原始標籤
                pred_regime_encoded = market_model.predict(features)[0]
                is_bull_market = (pred_regime_encoded == 1) # 1通常代表牛市
                
                # 統計熊市天數
                if not is_bull_market:
                    bear_market_days += 1
                
                # 添加調試信息，每1000天輸出一次市場情勢
                if equity_curve.index.get_loc(current_date) % 1000 == 0:
                    logger.info(f"日期: {current_date}, 市場情勢: {'牛市' if is_bull_market else '熊市'}, 預測編碼: {pred_regime_encoded}")

        # 2. 處理現有持倉 (平倉邏輯)
        # 迭代持倉副本，以避免在迭代過程中修改字典
        for stock_id in list(positions.keys()):
            pos = positions.get(stock_id)
            if not pos or stock_id not in all_stock_data or current_date not in all_stock_data[stock_id].index:
                continue

            exit_reason, row = None, all_stock_data[stock_id].loc[current_date]
            limit_up = row['limit_up']
            limit_down = row['limit_down']

            # === 新增最大虧損與停利判斷 ===
            if pos['direction'] == 'long':
                if float(row['low']) <= pos['stop_loss']:
                    exit_reason = 'StopLoss'
                elif float(row['close']) <= pos['entry_price'] * (1 - MAX_LOSS_PER_TRADE):
                    exit_reason = 'MaxLoss'
                elif float(row['close']) >= pos['entry_price'] * (1 + TAKE_PROFIT):
                    exit_reason = 'TakeProfit'
            elif pos['direction'] == 'short':
                if float(row['high']) >= pos['stop_loss']:
                    exit_reason = 'StopLoss'
                elif float(row['close']) >= pos['entry_price'] * (1 + MAX_LOSS_PER_TRADE):
                    exit_reason = 'MaxLoss'
                elif float(row['close']) <= pos['entry_price'] * (1 - TAKE_PROFIT):
                    exit_reason = 'TakeProfit'

            # === 移動停損 ===
            atr = row['atr_14'] if 'atr_14' in row else None
            current_price = row['close']
            if atr and not pd.isna(atr):
                if pos['direction'] == 'long':
                    pos['stop_loss'] = max(pos['stop_loss'], current_price - atr * ATR_MULTIPLIER)
                elif pos['direction'] == 'short':
                    pos['stop_loss'] = min(pos['stop_loss'], current_price + atr * ATR_MULTIPLIER)

            if exit_reason:
                # 平倉價格使用次日開盤價，如果沒有則使用當日收盤價
                exit_price = row['next_day_open']
                if pd.isna(exit_price) or exit_price <= 0: # 確保價格有效
                    exit_price = row['close']
                    logger.debug(f"{stock_id} {current_date}: next_day_open 無效，使用 close price {exit_price} 作為平倉價。")
                    if pd.isna(exit_price) or exit_price <= 0: # 如果收盤價也無效，則無法平倉
                        logger.warning(f"{stock_id} {current_date}: 無法找到有效平倉價格，跳過平倉。")
                        continue

                # 平倉時 - 修正：允許多頭和空頭交易平倉
                if (exit_price >= limit_down and exit_price <= limit_up):
                    # 可成交
                    pnl = 0
                    if pos['direction'] == 'long':
                        # 多頭平倉 (賣出)
                        pnl = (exit_price - pos['entry_price']) * pos['shares']
                        # 現金增加，扣除賣出手續費和交易稅
                        cash += exit_price * pos['shares'] * (1 - TRANSACTION_FEE_RATE - TRANSACTION_TAX_RATE)
                    else: # short
                        # 空頭平倉 (買回)
                        pnl = (pos['entry_price'] - exit_price) * pos['shares']
                        # 現金處理：需要支付買回成本 (含手續費)
                        # 空頭平倉時，需要從現金中扣除買回成本
                        buyback_cost = exit_price * pos['shares'] * (1 + TRANSACTION_FEE_RATE)
                        cash -= buyback_cost

                    trades_log.append({
                        'stock_id': stock_id,
                        'direction': pos['direction'],
                        'entry_date': pos['entry_date'],
                        'entry_price': pos['entry_price'],
                        'shares': pos['shares'],
                        'exit_date': current_date, # 修正：使用當前日期而不是加1天
                        'exit_price': exit_price,
                        'pnl_dollar_gross': pnl,
                        'exit_reason': exit_reason
                    })
                    del positions[stock_id] # 移除持倉
                else:
                    # 無法成交，持倉繼續
                    continue

        # 3. 考慮開新倉邏輯 (如果還有開倉額度)
        if len(positions) < MAX_OPEN_POSITIONS and not drawdown_triggered:
            for stock_id, df_feat in stock_features.items():
                if len(positions) >= MAX_OPEN_POSITIONS or stock_id in positions:
                    continue
                if current_date not in df_feat.index:
                    continue
                features_today = df_feat.loc[[current_date]][stock_feature_cols]
                if features_today.empty or features_today.isnull().any().any():
                    continue
                pred_probs = stock_model.predict_proba(features_today)[0]
                pred_label_encoded = np.argmax(pred_probs)
                pred_label = stock_label_encoder.inverse_transform([pred_label_encoded])[0]
                pred_confidence = pred_probs[pred_label_encoded]
                rsi_today = df_feat.loc[current_date, 'rsi_14'] if 'rsi_14' in df_feat.columns else 50

                # === 優化熊市做空條件 ===
                if is_bull_market:
                    should_trade_long = (pred_label == 1) and (pred_confidence > PREDICTION_PROB_THRESHOLD)
                    should_trade_short = False
                else:
                    should_trade_long = False
                    should_trade_short = (pred_label == -1) and (pred_confidence > PREDICTION_PROB_THRESHOLD)

                # === RSI輔助 ===
                if USE_RSI_FILTER:
                    if should_trade_long and rsi_today > RSI_LONG_THRESHOLD:
                        should_trade_long = False
                    if should_trade_short and rsi_today < RSI_SHORT_THRESHOLD:
                        should_trade_short = False

                if should_trade_long or should_trade_short:
                    entry_price, atr = df_feat.loc[current_date, ['next_day_open', 'atr_14']]

                    if pd.isna(entry_price) or entry_price <= 0 or pd.isna(atr) or atr <= 0:
                        logger.debug(f"{stock_id} {current_date}: 進場價格或ATR數據無效，跳過開倉。")
                        continue

                    # 計算部位大小 (基於波動性調整風險)
                    # 每筆交易風險敞口 = 組合總價值 * 固定比例 (例如1.5%)
                    # 部位大小 = (組合總價值 * 0.015) / (ATR * ATR_MULTIPLIER) / 每股價格
                    # 這裡假設每筆交易風險控制在總組合價值的1.5%左右
                    portfolio_value = cash + sum(p.get('value', 0) for p in positions.values()) # 包含現金和持倉市值
                    # 計算每單位風險的價值
                    risk_per_share = atr * ATR_MULTIPLIER
                    if risk_per_share <= 0: # 避免除以零或無效風險
                        continue
                    
                    # 計算股數 (以千股為單位取整)
                    # 總風險金額 / 每股風險金額 = 股數上限
                    # 這裡的 0.015 是每筆交易允許的最大資金佔比，應與風險管理策略的設計對應
                    raw_shares = (portfolio_value * 0.015) / risk_per_share
                    shares = int(raw_shares / entry_price / 1000) * 1000 # 轉換為股數並取千股整數
                    
                    if shares < 1000: # 最少購買1000股 (或設定一個更低的最小交易單位)
                        logger.debug(f"{stock_id} {current_date}: 計算股數 {shares} 低於最小交易單位，跳過開倉。")
                        continue

                    # === 單一股票最大持倉比例 ===
                    portfolio_value = cash + sum(p.get('value', 0) for p in positions.values())
                    max_position_value = portfolio_value * MAX_POSITION_PCT
                    # 計算現有持倉
                    current_position_value = 0
                    if stock_id in positions:
                        current_position_value = positions[stock_id]['shares'] * positions[stock_id]['entry_price']
                    # 檢查單一股票最大持倉
                    if (shares * entry_price + current_position_value) > max_position_value:
                        continue

                    if should_trade_long:
                        # 檢查現金是否足夠支付買入成本 (股票數量 * 進場價格 * (1 + 手續費率))
                        cost = shares * entry_price * (1 + TRANSACTION_FEE_RATE)
                        if cash >= cost:
                            stop_loss_price = entry_price - atr * ATR_MULTIPLIER
                            positions[stock_id] = {
                                'shares': shares,
                                'entry_price': entry_price,
                                'entry_date': current_date, # 修正：使用當前日期
                                'stop_loss': stop_loss_price,
                                'direction': 'long',
                                'value': 0 # 初始化持倉市值
                            }
                            cash -= cost
                            logger.debug(f"{stock_id} {current_date}: 開立多頭部位。股數: {shares}, 進場價: {entry_price:.2f}, 止損: {stop_loss_price:.2f}")
                        else:
                            logger.debug(f"{stock_id} {current_date}: 現金不足以開立多頭部位。需要: {cost:.2f}, 現金: {cash:.2f}")

                    elif should_trade_short:
                        # 空頭開倉：賣出股票，收到現金
                        # 賣出時立即收到現金，並扣除手續費和交易稅
                        revenue = shares * entry_price * (1 - TRANSACTION_FEE_RATE - TRANSACTION_TAX_RATE)
                        stop_loss_price = entry_price + atr * ATR_MULTIPLIER
                        positions[stock_id] = {
                            'shares': shares,
                            'entry_price': entry_price,
                            'entry_date': current_date, # 修正：使用當前日期
                            'stop_loss': stop_loss_price,
                            'direction': 'short',
                            'value': 0 # 初始化持倉市值
                        }
                        cash += revenue
                        short_executed += 1
                        logger.debug(f"{stock_id} {current_date}: 開立空頭部位。股數: {shares}, 進場價: {entry_price:.2f}, 止損: {stop_loss_price:.2f}")

        # 4. 更新每日權益曲線
        holdings_value = 0
        for stock_id, pos in positions.items():
            if current_date in all_stock_data[stock_id].index:
                current_price = all_stock_data[stock_id].loc[current_date, 'close']
            else:
                current_price = pos['entry_price']
            try:
                if pd.isna(current_price) or float(current_price) <= 0:
                    current_price = pos['entry_price']
            except (ValueError, TypeError):
                current_price = pos['entry_price']
            if pos.get('direction') == 'long':
                pos['value'] = pos['shares'] * current_price
            elif pos.get('direction') == 'short':
                pos['value'] = (pos['shares'] * pos['entry_price']) + (pos['entry_price'] - current_price) * pos['shares']
            holdings_value += pos.get('value', 0)
        equity_curve.loc[current_date] = cash + holdings_value

        # === 檢查組合最大回撤 ===
        peak = equity_curve[:current_date].max() if not pd.isna(equity_curve[:current_date]).all() else INITIAL_CAPITAL
        if peak > 0:
            drawdown = (equity_curve.loc[current_date] - peak) / peak
            if drawdown <= -MAX_DRAWDOWN_PCT and not drawdown_triggered:
                drawdown_triggered = True
                logger.warning(f"組合最大回撤超過{MAX_DRAWDOWN_PCT*100:.0f}%，{current_date} 強制平倉並停止新開倉！")
                # 強制平倉所有持倉
                for stock_id in list(positions.keys()):
                    pos = positions[stock_id]
                    row = all_stock_data[stock_id].loc[current_date]
                    exit_price = row['close']
                    pnl = (exit_price - pos['entry_price']) * pos['shares'] if pos['direction']=='long' else (pos['entry_price'] - exit_price) * pos['shares']
                    cash += exit_price * pos['shares'] * (1 - TRANSACTION_FEE_RATE - TRANSACTION_TAX_RATE) if pos['direction']=='long' else -exit_price * pos['shares'] * (1 + TRANSACTION_FEE_RATE)
                    trades_log.append({
                        'stock_id': stock_id,
                        'direction': pos['direction'],
                        'entry_date': pos['entry_date'],
                        'entry_price': pos['entry_price'],
                        'shares': pos['shares'],
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'pnl_dollar_gross': pnl,
                        'exit_reason': 'MaxDrawdownForceExit'
                    })
                    del positions[stock_id]

    logger.info("回測執行完畢，開始計算績效指標。")
    
    # 輸出統計信息
    total_days = len(equity_curve)
    logger.info(f"總回測天數: {total_days}")
    logger.info(f"熊市天數: {bear_market_days} ({bear_market_days/total_days*100:.2f}%)")
    logger.info(f"空頭交易嘗試次數: {short_attempts}")
    logger.info(f"空頭交易執行次數: {short_executed}")

    # 5. 計算績效指標 (KPIs)
    kpis = {}
    trades_df = pd.DataFrame(trades_log)
    valid_equity = equity_curve.dropna() # 移除回測期開始前或數據缺失導致的NaN值

    if not valid_equity.empty and len(valid_equity) > 1:
        final_equity = valid_equity.iloc[-1]
        initial_capital = INITIAL_CAPITAL
        returns = valid_equity.pct_change().dropna()
        
        # 計算總年數，確保至少為1年以避免除以0或過小的年數
        # 修正：使用pandas的日期索引進行計算
        total_years = (valid_equity.index[-1] - valid_equity.index[0]).days / 365.25  # type: ignore
        if total_years <= 0: total_years = 1 # 避免除以零

        kpis['最終權益'] = final_equity
        kpis['總淨利 ($)'] = final_equity - initial_capital
        kpis['總淨利 (%)'] = (kpis['總淨利 ($)'] / initial_capital) * 100

        # 年化報酬率 (CAGR)
        if initial_capital > 0 and final_equity > 0:
            kpis['年化報酬率 (%)'] = ((final_equity / initial_capital) ** (1 / total_years) - 1) * 100
        else:
            kpis['年化報酬率 (%)'] = 0.0

        # 最大回撤
        peak = valid_equity.expanding().max()
        drawdown = (valid_equity - peak) / peak
        kpis['最大回撤 (%)'] = drawdown.min() * 100 if not drawdown.empty else 0.0

        # 夏普比率 (假設無風險利率為0)
        if not returns.empty and returns.std() != 0:
            kpis['夏普比率'] = (returns.mean() / returns.std()) * np.sqrt(ANNUAL_TRADING_DAYS)
        else:
            kpis['夏普比率'] = 0.0

        # 卡瑪比率 (Calmar Ratio)
        if kpis.get('最大回撤 (%)', 0) < 0: # 只有在有回撤時計算
            kpis['卡瑪比率'] = kpis['年化報酬率 (%)'] / abs(kpis['最大回撤 (%)'])
        else:
            kpis['卡瑪比率'] = np.inf if kpis.get('年化報酬率 (%)', 0) > 0 else 0 # 無回撤但有收益為無限大，無收益為0
    else:
        logger.warning("權益曲線數據不足或無效，無法計算績效指標。")
        
    if not trades_df.empty:
        wins = trades_df[trades_df['pnl_dollar_gross'] > 0]
        losses = trades_df[trades_df['pnl_dollar_gross'] < 0]
        kpis['總交易次數'] = len(trades_df)
        kpis['獲利交易次數'] = len(wins)
        kpis['虧損交易次數'] = len(losses)
        kpis['勝率 (%)'] = (len(wins) / len(trades_df) * 100) if len(trades_df) > 0 else 0.0
        
        # 新增：多空交易統計
        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']
        kpis['多頭交易次數'] = len(long_trades)
        kpis['空頭交易次數'] = len(short_trades)
        kpis['多頭勝率 (%)'] = (len(long_trades[long_trades['pnl_dollar_gross'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0.0
        kpis['空頭勝率 (%)'] = (len(short_trades[short_trades['pnl_dollar_gross'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0.0

        # --- 新增：統計個股損益，繪製最賺/最虧個股K線圖 ---
        # 統計每檔個股總損益
        stock_pnl = trades_df.groupby('stock_id')['pnl_dollar_gross'].sum()
        best_stock = stock_pnl.idxmax()
        worst_stock = stock_pnl.idxmin()
        for stock_id, tag in [(best_stock, 'best'), (worst_stock, 'worst')]:
            df = all_stock_data[stock_id]
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(df.index, df['close'], label='收盤價', color='black')
            # 標註所有進出場點
            stock_trades = trades_df[trades_df['stock_id'] == stock_id]
            for _, row in stock_trades.iterrows():
                entry = row['entry_date']
                exit = row['exit_date']
                ax.axvline(entry, color='green', linestyle='--', alpha=0.5)
                ax.axvline(exit, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f"{stock_id} 回測K線圖（{tag}）")
            ax.set_ylabel('價格')
            ax.legend()
            plt.tight_layout()
            if 'report_dir' in locals():
                save_path = os.path.join(report_dir, f'{tag}_stock_{stock_id}_kline.png')
            else:
                save_path = f'{tag}_stock_{stock_id}_kline.png'
            plt.savefig(save_path, dpi=200)
            plt.close(fig)
    else:
        logger.warning("沒有產生任何交易，無法計算交易相關指標。")
        kpis['總交易次數'] = 0
        kpis['獲利交易次數'] = 0
        kpis['虧損交易次數'] = 0
        kpis['勝率 (%)'] = 0.0

    return BacktestResult(params={}, kpis=kpis, equity_curve=valid_equity, trades=trades_df)

def setup_plot_style():
    """
    設定Matplotlib圖表的顯示風格，包括中文字體。
    """
    try:
        # 嘗試設置支持中文的字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
    except Exception as e:
        logger.warning(f"警告：無法設定中文字體。請確保系統已安裝 'Microsoft JhengHei' 或 'Arial Unicode MS'。錯誤: {e}")

def plot_equity_curve(equity_curve, charts_dir, kpis):
    """
    繪製權益曲線圖並保存。
    """
    setup_plot_style() # 設定字體以支持中文顯示
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(equity_curve.index, equity_curve, label='權益曲線', color='blue', linewidth=1.5)
    
    # 格式化KPIs用於標題
    title_kpi_text = (f"年化報酬率: {kpis.get('年化報酬率 (%)', 0):.2f}% | "
                      f"最大回撤: {kpis.get('最大回撤 (%)', 0):.2f}% | "
                      f"夏普比率: {kpis.get('夏普比率', 0):.2f}")
    title = f'XGBoost 多空雙向策略權益曲線\n{title_kpi_text}'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('總權益 (NTD)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10) # 顯示圖例
    plt.tight_layout() # 自動調整佈局
    
    # 確保 charts_dir 存在
    os.makedirs(charts_dir, exist_ok=True)
    file_path = os.path.join(charts_dir, "xgboost_long_short_equity_curve.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight') # 高解析度保存
    plt.close(fig) # 關閉圖表以釋放記憶體

def format_kpis_for_display(kpis: Dict[str, Any]) -> str:
    """
    將KPIs字典格式化為易於閱讀的字串報告。
    """
    lines = ["\n" + "="*40, "   << 投資組合績效報告 - 詳細KPIs >>", "="*40]
    lines.append("\n--- 總體績效概覽 ---")
    lines.append(f"{'最終權益':<20}: {kpis.get('最終權益', 0):,.2f}")
    lines.append(f"{'淨利 ($)':<20}: {kpis.get('總淨利 ($)', 0):,.2f}")
    lines.append(f"{'淨利 (%)':<20}: {kpis.get('總淨利 (%)', 0):.2f}")
    lines.append(f"{'年化報酬率 (%)':<20}: {kpis.get('年化報酬率 (%)', 0):.2f}")
    lines.append(f"{'最大回撤 (%)':<20}: {kpis.get('最大回撤 (%)', 0):.2f}")
    lines.append(f"{'夏普比率':<20}: {kpis.get('夏普比率', 0):.2f}")
    lines.append(f"{'卡瑪比率':<20}: {kpis.get('卡瑪比率', 0):.2f}")
    lines.append("\n--- 交易行為分析 ---")
    lines.append(f"{'總交易次數':<20}: {kpis.get('總交易次數', 0)}")
    lines.append(f"{'獲利交易次數':<20}: {kpis.get('獲利交易次數', 0)}")
    lines.append(f"{'虧損交易次數':<20}: {kpis.get('虧損交易次數', 0)}")
    lines.append(f"{'勝率 (%)':<20}: {kpis.get('勝率 (%)', 0):.2f}")
    lines.append(f"{'多頭交易次數':<20}: {kpis.get('多頭交易次數', 0)}")
    lines.append(f"{'空頭交易次數':<20}: {kpis.get('空頭交易次數', 0)}")
    lines.append(f"{'多頭勝率 (%)':<20}: {kpis.get('多頭勝率 (%)', 0):.2f}")
    lines.append(f"{'空頭勝率 (%)':<20}: {kpis.get('空頭勝率 (%)', 0):.2f}")
    return "\n".join(lines)

def main():
    logger.info("===== XGBoost 多空整合策略回測開始 =====")
    try:
        # 載入預先訓練好的股票模型和市場情勢模型
        stock_model_data = joblib.load(STOCK_MODEL_PATH)
        regime_model = joblib.load(REGIME_MODEL_PATH)
        logger.info("兩個AI模型均載入成功。")
    except FileNotFoundError as e:
        logger.error(f"錯誤：找不到模型檔案。請確認您已運行模型訓練腳本並生成模型檔案。 - {e}"); return
    except Exception as e:
        logger.error(f"載入模型時發生未知錯誤: {e}"); return

    try:
        with open(LIST_PATH, "r", encoding="utf-8") as f:
            stock_list = sorted([l.strip() for l in f if l.strip().isdigit()])
    except FileNotFoundError:
        logger.error(f"錯誤：找不到股票列表檔案 {LIST_PATH}！請確認檔案路徑正確。"); return
    except Exception as e:
        logger.error(f"讀取股票列表時發生錯誤: {e}"); return
    
    # 載入所有股票的原始數據
    all_stock_data = {sid: df for sid in tqdm.tqdm(stock_list, desc="加載個股原始數據") if (df := prep_data(sid)) is not None}
    
    # 載入大盤數據並計算特徵
    market_raw_data = prep_data(MARKET_INDEX_ID)
    if market_raw_data is None:
        logger.error("無法加載大盤數據，程式終止。請檢查大盤數據檔案是否存在且有效。"); return
    market_features = calculate_features(market_raw_data, is_market=True)
    logger.info("所有數據加載完成。")

    # 執行回測
    result = run_xgboost_long_short_backtest(stock_model_data, regime_model, all_stock_data, market_features)
    
    # 處理回測結果
    if result and result.kpis:
        kpi_report_str = format_kpis_for_display(result.kpis)
        logger.info("\n===== XGBoost 多空策略回測結果 =====")
        logger.info(kpi_report_str)
        
        # 建立報告目錄
        report_dir = os.path.join(BASE_PATH, "runs", f"run_XGBoost_LS_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(report_dir, exist_ok=True)
        
        # 保存KPI報告到文件
        kpi_file_path = os.path.join(report_dir, "kpi_report.txt")
        with open(kpi_file_path, "w", encoding="utf-8") as f:
            f.write(kpi_report_str)
        logger.info(f"績效報告已儲存至: {kpi_file_path}")

        # 繪製並保存權益曲線圖
        if result.equity_curve is not None and not result.equity_curve.empty:
            plot_equity_curve(result.equity_curve, report_dir, result.kpis)
            logger.info(f"權益曲線圖已儲存至: {report_dir}")
        else:
            logger.warning("權益曲線數據為空，未能生成權益曲線圖。")
            
        # 保存交易紀錄 (如果有的話)
        if result.trades is not None and not result.trades.empty:
            trades_file_path = os.path.join(report_dir, "trades_log.csv")
            result.trades.to_csv(trades_file_path, encoding='utf-8-sig', index=False)
            logger.info(f"交易紀錄已儲存至: {trades_file_path}")
        else:
            logger.info("沒有產生任何交易紀錄。")
            
    else:
        logger.error("AI策略回測沒有產生任何結果或結果無效。")

    logger.info("===== XGBoost 多空整合策略回測結束 =====")

if __name__ == '__main__':
    main()