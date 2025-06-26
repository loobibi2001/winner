# 檔名: AI_Web_Dashboard.py (最終日期顯示修正版)
# =====================================================================
# 專案: 機器學習策略專案
# 階段: 最終成品 v4.4 - 修正日期顯示邏輯
#
# 本次更新:
#   - 核心修正: 動態調整分析結果的標題，明確顯示「數據日」與「建議日」。
#   - 介面優化: 增加數據來源時間戳，讓資訊更透明。
#   - 這是一個完整、獨立、可執行的最終成品腳本。
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import talib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# --- 基本設定與路徑 ---
st.set_page_config(page_title="雙AI智慧交易儀表板", layout="wide")

BASE_DIR = r"D:\飆股篩選\winner"
DATA_DIR = os.path.join(BASE_DIR, "StockData_Parquet")
STOCK_MODEL_PATH = os.path.join(BASE_DIR, "xgboost_long_short_model.joblib")
REGIME_MODEL_PATH = os.path.join(BASE_DIR, "regime_model.joblib")
LIST_PATH = os.path.join(BASE_DIR, "stock_list.txt")
MARKET_INDEX_ID = "TAIEX"
PORTFOLIO_PATH = os.path.join(BASE_DIR, "my_portfolio.csv")

# --- 快取(Cache)功能 ---
@st.cache_resource
def load_models():
    try:
        stock_model_data = joblib.load(STOCK_MODEL_PATH)
        regime_model = joblib.load(REGIME_MODEL_PATH)
        return stock_model_data, regime_model
    except FileNotFoundError as e:
        st.error(f"錯誤：找不到模型檔案 - {e}。")
        return None, None

@st.cache_data
def get_stock_list():
    try:
        with open(LIST_PATH, "r", encoding="utf-8") as f:
            return sorted([l.strip() for l in f if l.strip().isdigit()])
    except FileNotFoundError:
        st.error(f"錯誤：找不到股票列表檔案 {LIST_PATH}！")
        return []

@st.cache_data
def load_and_prep_data(stock_id):
    file_path = os.path.join(DATA_DIR, f"{stock_id}_history.parquet")
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_parquet(file_path); df.columns = [c.lower() for c in df.columns]
        if 'date' not in df.columns: df = df.reset_index()
        df['date'] = pd.to_datetime(df['date']); df = df.sort_values(by='date').set_index('date')
        required = ['high', 'low', 'close', 'open', 'volume']
        if df.empty or any(c not in df.columns for c in required) or df[required].isnull().any().any() or df['volume'].le(0).any(): return None
        df['next_day_open'] = df['open'].shift(-1)
        return df.dropna(subset=required + ['next_day_open'])
    except: return None

def calculate_features(df: pd.DataFrame, is_market: bool = False) -> pd.DataFrame:
    df_feat = df.copy()
    # 確保與模型訓練時的特徵列表一致
    market_feature_list = ['price_change_1d','price_change_5d','ma_20','ma20_slope','ma_60','rsi_14','macd','kd_k','atr_14','bollinger_width']
    stock_feature_list = market_feature_list + ['volume_change_1d', 'price_vs_ma20', 'williams_r', 'ma_5', 'macdsignal', 'macdhist', 'kd_d']
    
    df_feat['price_change_1d'] = df_feat['close'].pct_change(1)
    df_feat['price_change_5d'] = df_feat['close'].pct_change(5)
    df_feat['volume_change_1d'] = df_feat['volume'].pct_change(1)
    
    df_feat['ma_5'] = talib.SMA(df_feat['close'], 5)
    df_feat['ma_20'] = talib.SMA(df_feat['close'], 20)
    
    # 加入ma20_slope特徵計算
    df_feat['ma20_slope'] = df_feat['ma_20'].pct_change(1)
    
    df_feat['ma_60'] = talib.SMA(df_feat['close'], 60)
    df_feat['rsi_14'] = talib.RSI(df_feat['close'], 14)
    df_feat['macd'], df_feat['macdsignal'], df_feat['macdhist'] = talib.MACD(df_feat['close'], 12, 26, 9)
    df_feat['kd_k'], df_feat['kd_d'] = talib.STOCH(df_feat['high'], df_feat['low'], df_feat['close'], 9, 3, 3)
    df_feat['atr_14'] = talib.ATR(df_feat['high'], df_feat['low'], df_feat['close'], 14)
    upper, middle, lower = talib.BBANDS(df_feat['close'], 20)
    df_feat['bollinger_width'] = np.where(middle > 0, (upper - lower) / middle, 0)
    
    if not is_market:
        df_feat['price_vs_ma20'] = np.where(df_feat['ma_20'] > 0, df_feat['close'] / df_feat['ma_20'], 1)
        df_feat['williams_r'] = talib.WILLR(df_feat['high'], df_feat['low'], df_feat['close'], 14)
    
    final_features = market_feature_list if is_market else stock_feature_list
    return df_feat.replace([np.inf, -np.inf], np.nan).dropna(subset=final_features)

def load_portfolio():
    try:
        return pd.read_csv(PORTFOLIO_PATH, dtype={'股票代號': str})
    except FileNotFoundError:
        return pd.DataFrame([{"股票代號": "2330", "交易方向": "做多", "持有股數": 1000, "成本價": 950.0}, {"股票代號": "", "交易方向": "做多", "持有股數": 0, "成本價": 0.0}])

def save_portfolio(df: pd.DataFrame):
    df_to_save = df.dropna(subset=['股票代號']).copy(); df_to_save = df_to_save[df_to_save['股票代號'].astype(str).str.strip() != '']
    df_to_save.to_csv(PORTFOLIO_PATH, index=False, encoding='utf-8-sig')
    st.sidebar.success("持股已成功儲存！")

@st.cache_data(show_spinner=False)
def plot_stock_chart_with_signal(stock_id, signal_date, signal_type, entry_price):
    df = load_and_prep_data(stock_id)
    if df is None: return None
    actual_signal_date = df.index.asof(pd.to_datetime(signal_date))
    if pd.isna(actual_signal_date): return None
    start_date = actual_signal_date - timedelta(days=120); end_date = actual_signal_date + timedelta(days=30)
    df_plot = df.loc[start_date:end_date].copy()
    if df_plot.empty: return None
    df_plot['ma_20'] = talib.SMA(df_plot['close'], 20); df_plot['ma_60'] = talib.SMA(df_plot['close'], 60)
    fig, ax = plt.subplots(figsize=(15, 7))
    for idx, row in df_plot.iterrows():
        color = 'red' if row['close'] >= row['open'] else 'green'
        ax.add_patch(Rectangle((mdates.date2num(idx)-0.2, row['open']), 0.4, row['close']-row['open'], facecolor=color, edgecolor=color, zorder=2))
        ax.plot([mdates.date2num(idx), mdates.date2num(idx)], [row['low'], row['high']], color=color, linewidth=0.5, zorder=1)
    ax.plot(df_plot.index, df_plot['ma_20'], color='blue', linewidth=1, label='20日均線'); ax.plot(df_plot.index, df_plot['ma_60'], color='orange', linewidth=1, label='60日均線', linestyle='--')
    signal_price_y_position = df.loc[actual_signal_date, 'low'] * 0.98 if signal_type == '做多' else df.loc[actual_signal_date, 'high'] * 1.02
    marker = '^' if signal_type == '做多' else 'v'; color = 'fuchsia' if signal_type == '做多' else 'cyan'
    label = 'AI做多訊號' if signal_type == '做多' else 'AI放空訊號'
    ax.scatter(actual_signal_date, signal_price_y_position, marker=marker, color=color, s=200, zorder=3, label=label)
    ax.annotate(f"明日開盤參考價: {entry_price:.2f}", xy=(actual_signal_date, signal_price_y_position), xytext=(15, 15), textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"), fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    ax.set_title(f"{stock_id} K線圖與AI訊號分析", fontsize=16); ax.set_ylabel("價格"); ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); fig.autofmt_xdate()
    return fig

def get_trading_advice(portfolio_df, prob_threshold):
    stock_model_data, regime_model = load_models()
    if not stock_model_data or not regime_model: return "模型載入失敗", None, None, [], []
    stock_model, label_encoder = stock_model_data['model'], stock_model_data['label_encoder']
    portfolio = {str(row['股票代號']).strip(): {'direction': row['交易方向']} for index, row in portfolio_df.iterrows() if pd.notna(row['股票代號']) and str(row['股票代號']).strip() and pd.notna(row['交易方向'])}
    market_data = load_and_prep_data(MARKET_INDEX_ID)
    if market_data is None or market_data.empty: return "無法獲取大盤數據", None, None, [], []
    # 取最新一筆資料的日期
    latest_date = market_data.index.max()
    next_trading_day = latest_date + pd.tseries.offsets.BDay(1)
    market_features = calculate_features(market_data, is_market=True)
    is_bull_market = True
    if latest_date in market_features.index:
        features = market_features.loc[[latest_date]][regime_model.feature_names_in_]
        if not features.empty: is_bull_market = regime_model.predict(features)[0] == 1
    regime_status = "☀️ 多頭市場 (適宜做多)" if is_bull_market else "🌧️ 空頭市場 (適宜做空或觀望)"
    buy_signals, sell_signals = [], []
    all_stocks = get_stock_list()
    progress_bar = st.progress(0, text="AI模型正在分析所有股票...")
    for i, stock_id in enumerate(all_stocks):
        df = load_and_prep_data(stock_id)
        if df is None or df.empty or df.index.max() < latest_date - timedelta(days=5): continue
        df_feat = calculate_features(df)
        if latest_date not in df_feat.index: continue
        features_today = df_feat.loc[[latest_date]][stock_model.feature_names_in_]
        if features_today.empty: continue
        pred_probs = stock_model.predict_proba(features_today)[0]
        pred_label_encoded = np.argmax(pred_probs)
        pred_label = label_encoder.inverse_transform([pred_label_encoded])[0]
        if stock_id in portfolio:
            pos = portfolio[stock_id]
            if (pos['direction'] == '做多' and (not is_bull_market or pred_label != 1)): sell_signals.append(f"【平倉多單】 {stock_id}: 市場轉空或AI看多訊號消失。")
            elif (pos['direction'] == '放空' and (is_bull_market or pred_label != -1)): sell_signals.append(f"【回補空單】 {stock_id}: 市場轉多或AI看空訊號消失。")
        elif len(buy_signals) < 20: 
            pred_confidence = pred_probs[pred_label_encoded]
            entry_price = df_feat.loc[latest_date, 'next_day_open']
            if pd.notna(entry_price):
                if is_bull_market and pred_label == 1 and pred_confidence > prob_threshold: buy_signals.append({'stock_id': stock_id, 'signal': '做多', 'confidence': f"{pred_confidence:.0%}", 'entry_price': entry_price})
                elif not is_bull_market and pred_label == -1 and pred_confidence > prob_threshold: buy_signals.append({'stock_id': stock_id, 'signal': '放空', 'confidence': f"{pred_confidence:.0%}", 'entry_price': entry_price})
        progress_bar.progress((i + 1) / len(all_stocks))
    progress_bar.empty()
    return regime_status, latest_date.date(), next_trading_day.date(), buy_signals, sell_signals

# --- Streamlit 網頁介面 ---
st.title("📈 雙AI智慧交易儀表板")
st.caption(f"台灣時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with st.sidebar:
    st.header("⚙️ 設定與操作")
    st.subheader("我的多空持股")
    portfolio_df = load_portfolio()
    edited_portfolio = st.data_editor(portfolio_df, column_config={"股票代號": st.column_config.TextColumn("股票代號", required=True), "交易方向": st.column_config.SelectboxColumn("交易方向", options=["做多", "放空"], required=True), "持有股數": st.column_config.NumberColumn("持有股數", min_value=0, format="%d"), "成本價": st.column_config.NumberColumn("成本價", min_value=0.0, format="%.2f")}, num_rows="dynamic", key="portfolio_editor")
    if st.button("💾 儲存持股變更"): save_portfolio(edited_portfolio)
    st.markdown("---")
    prob_threshold = st.slider("AI 信心門檻", 0.40, 0.90, 0.55, 0.05, help="只有當AI預測的機率高於此門檻時，才會產生進場訊號。")
    run_button = st.button("🚀 開始分析")
    st.markdown("---")
    # 新增回測績效報告區塊
    st.subheader("📊 最近一次回測績效報告")
    try:
        with open("runs/run_XGBoost_LS_20250626_190014/kpi_report.txt", "r", encoding="utf-8") as f:
            kpi_report = f.read()
        st.text(kpi_report)
    except Exception as e:
        st.info("找不到回測績效報告，請先執行回測腳本。")

if 'buy_signals' not in st.session_state: st.session_state.buy_signals = []

if run_button:
    with st.spinner('AI大腦正在高速運轉中，請稍候...'):
        regime, analysis_date, next_day, buys, sells = get_trading_advice(edited_portfolio, prob_threshold)
    st.session_state.analysis_date = analysis_date; st.session_state.next_trading_day = next_day
    st.session_state.buy_signals = buys; st.session_state.sell_signals = sells; st.session_state.regime = regime

if 'regime' in st.session_state and st.session_state.regime:
    # --- 核心修改：動態標題 ---
    st.header(f"🗓️ {st.session_state.analysis_date} 收盤數據分析 (提供 {st.session_state.next_trading_day} 操作建議)")
    
    if "多頭" in st.session_state.regime:
        st.success(f"**當前大盤狀態: {st.session_state.regime}**")
        st.info("目前僅推薦做多機會，因為市場為多頭。")
    else:
        st.error(f"**當前大盤狀態: {st.session_state.regime}**")
        st.info("目前僅推薦放空機會，因為市場為空頭。")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 新機會建議")
        if st.session_state.buy_signals:
            for signal in st.session_state.buy_signals:
                st.metric(label=f"【建議{signal['signal']}】 {signal['stock_id']}", value=f"{signal['entry_price']:.2f}", delta=f"信心指數: {signal['confidence']}", delta_color="off")
        else: st.info("今日無新的進場訊號。")
    with col2:
        st.subheader("🔴 持股檢查建議")
        if st.session_state.sell_signals:
            for signal in st.session_state.sell_signals: st.warning(signal)
        else: st.info("您目前持股無需立即變動。")
    
    st.markdown("---")
    # 新增即時分析KPI區塊
    st.subheader("📈 今日即時AI分析KPI")
    st.write(f"今日建議做多數量：{sum(1 for s in st.session_state.buy_signals if s['signal']=='做多')}")
    st.write(f"今日建議放空數量：{sum(1 for s in st.session_state.buy_signals if s['signal']=='放空')}")
    st.write(f"今日建議平倉/回補數量：{len(st.session_state.sell_signals)}")
    
    st.header("🔍 訊號視覺化分析")
    if st.session_state.buy_signals:
        options = {f"{s['stock_id']} ({s['signal']})": s for s in st.session_state.buy_signals}
        selected_option = st.selectbox("請選擇一檔股票來查看K線圖與訊號點：", options=options.keys())
        if selected_option:
            selected_signal_info = options[selected_option]
            with st.spinner(f"正在繪製 {selected_signal_info['stock_id']} 的K線圖..."):
                fig = plot_stock_chart_with_signal(selected_signal_info['stock_id'], st.session_state.analysis_date, selected_signal_info['signal'], selected_signal_info['entry_price'])
                if fig: st.pyplot(fig)
                else: st.error("無法繪製此股票的圖表，可能是因為今天不是該股的交易日。")
    else:
        st.info("目前沒有新的交易建議可供分析。")
else:
    st.info("請在左方側邊欄編輯您的多空持股清單，點擊「儲存持股變更」以保存，然後點擊「開始分析」。")