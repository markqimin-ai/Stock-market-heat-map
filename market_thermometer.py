import akshare as ak
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- 页面配置 ----------------
st.set_page_config(page_title="A股市场温度计", layout="wide")
st.title("📊 A股市场评估工具 (基于霍华德·马克斯周期理论)")
st.caption("本工具基于上传的市场评估要点，通过实时量化数据表征市场热度。")


# ---------------- 工具函数 ----------------
def _pick_col(df: pd.DataFrame, candidates):
    cols = set(df.columns)
    for name in candidates:
        if name in cols:
            return name
    for name in candidates:
        for c in df.columns:
            if isinstance(c, str) and name.lower() in c.lower():
                return c
    raise KeyError(f"未找到列，候选: {candidates}，实际列: {list(df.columns)}")


def _rank_score(series, current, invert=False, window_days=None, dates=None):
    """计算 current 在 series 历史序列中的百分位 (0-100)。
    invert=True 时反转（值越小越热）。window_days 用于限定历史窗口。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if window_days is not None and dates is not None:
        cutoff = pd.to_datetime(dates).max() - pd.Timedelta(days=window_days)
        mask = pd.to_datetime(dates) >= cutoff
        s = s[mask]
    if len(s) == 0 or pd.isna(current):
        return float("nan")
    rank = (s <= current).mean() * 100.0
    return 100.0 - rank if invert else rank


def _fmt(v, fmt=".2f", unit=""):
    return f"{v:{fmt}}{unit}" if not np.isnan(v) else "N/A"


def _diag(pct, hot_label="偏热", cold_label="偏冷", mid_label="中性"):
    if np.isnan(pct):
        return "—"
    if pct >= 70:
        return hot_label
    if pct < 30:
        return cold_label
    return mid_label


def _diag_threshold(value, hot_threshold, cold_threshold, hot_label, cold_label, mid_label="中性", hot_is_high=True):
    """按固定阈值诊断（非分位）。
    hot_is_high=True: value >= hot_threshold 触发 hot_label
    hot_is_high=False: value <= cold_threshold 触发 hot_label（反转语义）"""
    if value is None or pd.isna(value):
        return "—"
    if hot_is_high:
        if value >= hot_threshold:
            return hot_label
        if value < cold_threshold:
            return cold_label
        return mid_label
    else:
        if value <= hot_threshold:
            return hot_label
        if value > cold_threshold:
            return cold_label
        return mid_label


# ---------------- 数据抓取 ----------------
@st.cache_data(ttl=3600, show_spinner="正在拉取市场数据…")
def fetch_market_data():
    """返回各维度指标字典。errors 收集失败项。"""
    data = {"errors": []}

    def _try(name, fn):
        try:
            return fn()
        except Exception as e:
            data["errors"].append(f"{name}: {e}")
            return None

    # ===== 走势/价格维度 =====
    # 全市场 PE (TTM 中位数) + 沪深300 收盘价
    pe_df = _try("全市场PE", lambda: ak.stock_a_ttm_lyr())
    if pe_df is not None:
        pe_df["date"] = pd.to_datetime(pe_df["date"])
        pe_df = pe_df.sort_values("date")
        latest = pe_df.iloc[-1]
        data["pe"] = float(latest["middlePETTM"])
        data["pe_pct"] = float(latest["quantileInRecent10YearsMiddlePeTtm"]) * 100
        data["pe_date"] = latest["date"].strftime("%Y-%m-%d")
        # 用 close 作为上证替代（全市场收盘价）
        data["index_close"] = float(latest["close"])
        close_series = pd.to_numeric(pe_df["close"], errors="coerce")
        ma30 = close_series.tail(30).mean()
        data["index_ma30"] = float(ma30)
        data["index_above_ma30"] = data["index_close"] > ma30

    # 融资融券余额
    margin_df = _try("融资融券", lambda: ak.macro_china_market_margin_sh())
    if margin_df is not None:
        col = _pick_col(margin_df, ["融资余额", "rzye"])
        date_col = _pick_col(margin_df, ["日期", "date", "opdate"])
        margin_df[date_col] = pd.to_datetime(margin_df[date_col])
        margin_df = margin_df.sort_values(date_col)
        series = pd.to_numeric(margin_df[col], errors="coerce")
        data["margin"] = float(series.iloc[-1])
        data["margin_pct"] = _rank_score(series, data["margin"], window_days=365*5,
                                          dates=margin_df[date_col])
        data["margin_date"] = margin_df[date_col].iloc[-1].strftime("%Y-%m-%d")

    # ===== 资金维度 =====
    # 10Y 国债收益率
    bond_df = _try("国债收益率", lambda: ak.bond_zh_us_rate())
    if bond_df is not None:
        col = _pick_col(bond_df, ["中国国债收益率10年", "10年", "q_10y"])
        date_col = _pick_col(bond_df, ["日期", "date"])
        bond_df[date_col] = pd.to_datetime(bond_df[date_col])
        bond_df = bond_df.sort_values(date_col)
        series = pd.to_numeric(bond_df[col], errors="coerce").dropna()
        data["yield_10y"] = float(series.iloc[-1])
        data["yield_pct"] = _rank_score(series, data["yield_10y"], invert=True,
                                         window_days=365*10, dates=bond_df[date_col])
        data["yield_date"] = bond_df[date_col].iloc[-1].strftime("%Y-%m-%d")

    # M2 同比 & M1 同比（剪刀差）
    m2_df = _try("M2货币供应", lambda: ak.macro_china_money_supply())
    if m2_df is not None:
        m2_col = _pick_col(m2_df, ["货币和准货币(M2)-同比增长"])
        m1_col = _pick_col(m2_df, ["货币(M1)-同比增长"])
        date_col = _pick_col(m2_df, ["月份", "date", "日期"])
        m2_df = m2_df.sort_values(date_col)
        data["m2_yoy"] = float(pd.to_numeric(m2_df[m2_col], errors="coerce").iloc[-1])
        data["m1_yoy"] = float(pd.to_numeric(m2_df[m1_col], errors="coerce").iloc[-1])
        data["m1_m2_scissor"] = data["m1_yoy"] - data["m2_yoy"]
        data["money_date"] = str(m2_df[date_col].iloc[-1])

    # 社融规模增量
    sf_df = _try("社会融资规模", lambda: ak.macro_china_shrzgm())
    if sf_df is not None:
        col = _pick_col(sf_df, ["社会融资规模增量"])
        date_col = _pick_col(sf_df, ["月份", "date", "日期"])
        sf_df = sf_df.sort_values(date_col)
        series = pd.to_numeric(sf_df[col], errors="coerce").dropna()
        if len(series) > 0:
            data["social_finance"] = float(series.iloc[-1])
            recent3 = series.tail(3).mean()
            prev3 = series.tail(6).head(3).mean() if len(series) >= 6 else float("nan")
            if not np.isnan(prev3):
                data["social_finance_trend"] = "回升" if recent3 > prev3 else "下行"
            else:
                data["social_finance_trend"] = "—"
            data["social_finance_date"] = str(sf_df[date_col].iloc[-1])
        else:
            data["social_finance"] = float("nan")
            data["social_finance_trend"] = "—"
            data["social_finance_date"] = str(sf_df[date_col].iloc[-1])

    # Shibor 隔夜
    shibor_df = _try("Shibor", lambda: ak.rate_interbank(
        market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="隔夜"))
    if shibor_df is not None:
        col = _pick_col(shibor_df, ["利率"])
        date_col = _pick_col(shibor_df, ["报告日", "日期", "date"])
        shibor_df[date_col] = pd.to_datetime(shibor_df[date_col])
        shibor_df = shibor_df.sort_values(date_col)
        series = pd.to_numeric(shibor_df[col], errors="coerce")
        data["shibor_overnight"] = float(series.iloc[-1])
        data["shibor_pct"] = _rank_score(series, data["shibor_overnight"],
                                          window_days=365*3, dates=shibor_df[date_col])
        data["shibor_date"] = shibor_df[date_col].iloc[-1].strftime("%Y-%m-%d")

    # LPR
    lpr_df = _try("LPR", lambda: ak.macro_china_lpr())
    if lpr_df is not None:
        data["lpr_1y"] = float(lpr_df["LPR1Y"].iloc[-1])
        data["lpr_5y"] = float(lpr_df["LPR5Y"].iloc[-1])
        # 是否较上次下调
        if len(lpr_df) >= 2:
            delta = lpr_df["LPR1Y"].iloc[-1] - lpr_df["LPR1Y"].iloc[-2]
            data["lpr_trend"] = "下调" if delta < 0 else ("上调" if delta > 0 else "持平")
        else:
            data["lpr_trend"] = "—"
        data["lpr_date"] = str(lpr_df["TRADE_DATE"].iloc[-1])

    # ===== 宏观维度 =====
    gdp_df = _try("GDP", lambda: ak.macro_china_gdp())
    if gdp_df is not None:
        col = _pick_col(gdp_df, ["国内生产总值-同比增长"])
        date_col = _pick_col(gdp_df, ["季度", "date", "日期"])
        gdp_df = gdp_df.sort_values(date_col)
        data["gdp_yoy"] = float(pd.to_numeric(gdp_df[col], errors="coerce").iloc[-1])
        data["gdp_date"] = str(gdp_df[date_col].iloc[-1])

    pmi_df = _try("PMI", lambda: ak.macro_china_pmi())
    if pmi_df is not None:
        mfg_col = _pick_col(pmi_df, ["制造业-指数"])
        nonmfg_col = _pick_col(pmi_df, ["非制造业-指数"])
        date_col = _pick_col(pmi_df, ["月份", "date", "日期"])
        pmi_df = pmi_df.sort_values(date_col)
        data["pmi_mfg"] = float(pd.to_numeric(pmi_df[mfg_col], errors="coerce").iloc[-1])
        data["pmi_nonmfg"] = float(pd.to_numeric(pmi_df[nonmfg_col], errors="coerce").iloc[-1])
        data["pmi_date"] = str(pmi_df[date_col].iloc[-1])

    ip_df = _try("工业增加值", lambda: ak.macro_china_industrial_production_yoy())
    if ip_df is not None:
        col = _pick_col(ip_df, ["今值"])
        date_col = _pick_col(ip_df, ["日期", "date"])
        ip_df = ip_df.sort_values(date_col)
        series = pd.to_numeric(ip_df[col], errors="coerce").dropna()
        if len(series) > 0:
            data["industrial_yoy"] = float(series.iloc[-1])
            valid_idx = series.index[-1]
            data["industrial_date"] = str(ip_df[date_col].iloc[valid_idx])
        else:
            data["industrial_yoy"] = float("nan")
            data["industrial_date"] = str(ip_df[date_col].iloc[-1])

    retail_df = _try("社消零售", lambda: ak.macro_china_consumer_goods_retail())
    if retail_df is not None:
        col = _pick_col(retail_df, ["同比增长"])
        date_col = _pick_col(retail_df, ["月份", "date", "日期"])
        retail_df = retail_df.sort_values(date_col)
        data["retail_yoy"] = float(pd.to_numeric(retail_df[col], errors="coerce").iloc[-1])
        data["retail_date"] = str(retail_df[date_col].iloc[-1])

    # ===== 情绪维度 =====
    # 北向资金（陆股通）— 近 20 日净流入
    hsgt_df = _try("北向资金", lambda: ak.stock_hsgt_hist_em(symbol="北向资金"))
    if hsgt_df is not None:
        col = _pick_col(hsgt_df, ["当日成交净买额"])
        date_col = _pick_col(hsgt_df, ["日期", "date"])
        hsgt_df[date_col] = pd.to_datetime(hsgt_df[date_col])
        hsgt_df = hsgt_df.sort_values(date_col)
        series = pd.to_numeric(hsgt_df[col], errors="coerce").dropna()
        if len(series) >= 20:
            recent20 = series.tail(20)
            data["northbound_20d_net"] = float(recent20.sum())
            data["northboard_trend"] = "净流入" if recent20.sum() > 0 else "净流出"
        data["northboard_date"] = hsgt_df[date_col].iloc[-1].strftime("%Y-%m-%d")

    # 两市成交额（国证A指，399317）
    all_cni_df = _try("国证A指", lambda: ak.index_all_cni())
    if all_cni_df is not None:
        gz_a = all_cni_df[all_cni_df["指数简称"] == "国证A指"]
        if len(gz_a) > 0:
            amount = float(pd.to_numeric(gz_a["成交额"].iloc[0], errors="coerce"))
            data["market_volume"] = amount
            data["market_volume_trend"] = "放量" if amount > 10000 else "缩量"
            data["market_volume_date"] = datetime.now().strftime("%Y-%m-%d")

    # 房价（百城）
    house_df = _try("百城房价", lambda: ak.macro_china_real_estate())
    if house_df is not None:
        col = _pick_col(house_df, ["涨跌幅"])
        date_col = _pick_col(house_df, ["日期", "date"])
        house_df = house_df.sort_values(date_col)
        data["house_price_yoy"] = float(pd.to_numeric(house_df[col], errors="coerce").iloc[-1])
        data["house_date"] = str(house_df[date_col].iloc[-1])

    # ===== 风险维度 =====
    # QVIX (中国版VIX，50ETF期权隐含波动率)
    qvix_df = _try("QVIX", lambda: ak.index_option_50etf_qvix())
    if qvix_df is not None:
        col = _pick_col(qvix_df, ["close"])
        date_col = _pick_col(qvix_df, ["date", "日期"])
        qvix_df[date_col] = pd.to_datetime(qvix_df[date_col])
        qvix_df = qvix_df.sort_values(date_col)
        series = pd.to_numeric(qvix_df[col], errors="coerce")
        data["qvix"] = float(series.iloc[-1])
        data["qvix_pct"] = _rank_score(series, data["qvix"], window_days=365*3,
                                        dates=qvix_df[date_col])
        data["qvix_date"] = qvix_df[date_col].iloc[-1].strftime("%Y-%m-%d")

    # USD/CNY 汇率 (BOC 中间价单位是"100美元换人民币"，需除100)
    fx_df = _try("汇率", lambda: ak.currency_boc_safe())
    if fx_df is not None:
        col = _pick_col(fx_df, ["美元"])
        date_col = _pick_col(fx_df, ["日期", "date"])
        fx_df[date_col] = pd.to_datetime(fx_df[date_col])
        fx_df = fx_df.sort_values(date_col)
        series = pd.to_numeric(fx_df[col], errors="coerce").dropna() / 100.0
        data["usd_cny"] = float(series.iloc[-1])
        # 近20日升值/贬值
        if len(series) >= 21:
            chg = (series.iloc[-1] / series.iloc[-21] - 1) * 100
            data["fx_20d_chg"] = float(chg)
            data["fx_trend"] = "贬值" if chg > 0 else "升值"
        data["fx_date"] = fx_df[date_col].iloc[-1].strftime("%Y-%m-%d")

    return data


data = fetch_market_data()

# ---------------- 综合热度（基于原三项核心指标） ----------------
parts, weights = [], []
for key, w in [("pe_pct", 0.4), ("margin_pct", 0.3), ("yield_pct", 0.3)]:
    v = data.get(key, float("nan"))
    if not (v is None or np.isnan(v)):
        parts.append(v * w)
        weights.append(w)
heat_score = sum(parts) / sum(weights) if weights else 50.0
heat_score = float(np.clip(heat_score, 0, 100))

# ---------------- 温度计 ----------------
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=heat_score,
    title={"text": "市场综合热度 (0-100)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "darkblue"},
        "steps": [
            {"range": [0, 30], "color": "lightgreen"},
            {"range": [30, 70], "color": "khaki"},
            {"range": [70, 100], "color": "crimson"},
        ],
        "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
    },
))
st.plotly_chart(fig, use_container_width=True)

# ---------------- 核心诊断看板 ----------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("资产价格水平 (全市场PE中位数)", _fmt(data.get("pe")), f"近10年分位: {_fmt(data.get('pe_pct'), '.2f', '%')}")
    if not np.isnan(data.get("pe_pct", float("nan"))):
        st.write(f"当前资产价格**{_diag(data['pe_pct'], '高', '低')}**")
with col2:
    if not np.isnan(data.get("margin", float("nan"))):
        st.metric("投资人情绪 (上证融资余额)", f"{data['margin'] / 1e8:.0f} 亿", f"近5年分位: {data['margin_pct']:.2f}%")
    else:
        st.metric("投资人情绪 (上证融资余额)", "N/A")
    st.write("反映投资人是否**乐观自信、渴望买进**")
with col3:
    if not np.isnan(data.get("yield_10y", float("nan"))):
        st.metric("利率水平 (10Y国债)", f"{data['yield_10y']:.2f}%", f"宽松度分位: {data['yield_pct']:.2f}%")
    else:
        st.metric("利率水平 (10Y国债)", "N/A")
    st.write("反映资本环境是**宽松**还是**紧缩**")

# ---------------- 具体指标明细表（核心三项） ----------------
st.subheader("📋 核心指标明细")
detail_rows = [
    {"维度": "资产价格", "指标": "全市场PE(TTM)中位数", "当前值": _fmt(data.get("pe")),
     "历史分位": _fmt(data.get("pe_pct"), ".2f", "%"), "权重": "40%",
     "加权贡献": _fmt(data.get("pe_pct", 0) * 0.4),
     "诊断": _diag(data.get("pe_pct", float("nan")), "估值偏高", "估值偏低")},
    {"维度": "投资人情绪", "指标": "上证融资余额", "当前值": _fmt(data.get("margin") / 1e8 if data.get("margin") else float("nan"), ".0f", " 亿"),
     "历史分位": _fmt(data.get("margin_pct"), ".2f", "%"), "权重": "30%",
     "加权贡献": _fmt(data.get("margin_pct", 0) * 0.3),
     "诊断": _diag(data.get("margin_pct", float("nan")), "情绪亢奋", "情绪低迷")},
    {"维度": "资本环境", "指标": "10年期国债收益率", "当前值": _fmt(data.get("yield_10y"), ".2f", "%"),
     "历史分位": _fmt(data.get("yield_pct"), ".2f", "%"), "权重": "30%",
     "加权贡献": _fmt(data.get("yield_pct", 0) * 0.3),
     "诊断": _diag(data.get("yield_pct", float("nan")), "资金宽松", "资金偏紧")},
    {"维度": "合计", "指标": "—", "当前值": "—", "历史分位": "—", "权重": "100%",
     "加权贡献": f"{heat_score:.2f}", "诊断": _diag(heat_score, "偏热", "偏冷")},
]
st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
st.caption("注：历史分位 = 当前值在过去 N 年样本中的百分位（PE/国债取近10年，融资余额取近5年）；诊断阈值：≥70% 偏热，<30% 偏冷。")

# ==================== 完整指标体系 ====================
st.divider()
st.subheader("📚 完整市场评估指标体系")
st.caption("下表覆盖宏观/资金/走势/情绪/风险 5 大维度共 20 项指标，按用户提供的霍华德·马克斯周期框架整理。可实时拉取的指标已自动诊断。")

# 指标体系表（含：维度/指标/数据源/积极区间/消极区间/解读/当前值/诊断）
def _threshold_diag(spec):
    """根据指标规范返回 (当前值, 诊断)"""
    key = spec.get("key")
    val = data.get(key) if key else None
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—", "—"
    if spec.get("diag_type") == "threshold":
        d = _diag_threshold(val, spec["hot"], spec["cold"], spec["hot_label"],
                            spec["cold_label"], hot_is_high=spec.get("hot_is_high", True))
    elif spec.get("diag_type") == "percentile":
        pct = data.get(spec["pct_key"], float("nan"))
        d = _diag(pct, spec["hot_label"], spec["cold_label"])
    elif spec.get("diag_type") == "custom":
        d = spec["fn"](val, data)
    else:
        d = "—"
    return spec["fmt"](val), d


def _yes_no(v, yes_label="是", no_label="否"):
    return yes_label if v else no_label


indicators = [
    # 宏观
    {"维度": "宏观", "核心指标": "GDP 同比", "数据源": "国家统计局",
     "积极区间": ">5%", "消极区间": "<3%", "信号解读": "衡量经济总量增长动能，<3%通常意味着需强力政策刺激。",
     "key": "gdp_yoy", "fmt": lambda v: f"{v:.2f}%",
     "diag_type": "threshold", "hot": 5, "cold": 3, "hot_label": "扩张", "cold_label": "收缩", "hot_is_high": True},

    {"维度": "宏观", "核心指标": "PMI (制造业)", "数据源": "统计局",
     "积极区间": ">50 (荣枯线上)", "消极区间": "<50 (荣枯线下)", "信号解读": "环比动能指标，>50代表扩张，连续低于50代表收缩。",
     "key": "pmi_mfg", "fmt": lambda v: f"{v:.2f}",
     "diag_type": "threshold", "hot": 50, "cold": 50, "hot_label": "扩张", "cold_label": "收缩", "hot_is_high": True},

    {"维度": "宏观", "核心指标": "PMI (非制造业)", "数据源": "统计局",
     "积极区间": ">50", "消极区间": "<50", "信号解读": "非制造业（含服务业/建筑业）景气度。",
     "key": "pmi_nonmfg", "fmt": lambda v: f"{v:.2f}",
     "diag_type": "threshold", "hot": 50, "cold": 50, "hot_label": "扩张", "cold_label": "收缩", "hot_is_high": True},

    {"维度": "宏观", "核心指标": "工业增加值同比", "数据源": "国家统计局",
     "积极区间": ">6%", "消极区间": "<4%", "信号解读": "反映生产端活跃度，过低暗示供应链或需求端疲软。",
     "key": "industrial_yoy", "fmt": lambda v: f"{v:.2f}%",
     "diag_type": "threshold", "hot": 6, "cold": 4, "hot_label": "活跃", "cold_label": "疲软", "hot_is_high": True},

    {"维度": "宏观", "核心指标": "社消零售总额同比", "数据源": "国家统计局",
     "积极区间": ">6%", "消极区间": "<4%", "信号解读": "反映内需消费能力，是经济转型的核心观测点。",
     "key": "retail_yoy", "fmt": lambda v: f"{v:.2f}%",
     "diag_type": "threshold", "hot": 6, "cold": 4, "hot_label": "旺盛", "cold_label": "疲软", "hot_is_high": True},

    # 资金
    {"维度": "资金", "核心指标": "社融规模存量增速", "数据源": "人行/统计局",
     "积极区间": "企稳回升 / 高于名义GDP", "消极区间": "持续下行", "信号解读": "实体经济的融资需求，反映未来经济活动的潜能。",
     "key": "social_finance", "fmt": lambda v: f"{v:.0f} 亿",
     "diag_type": "custom", "fn": lambda v, d: d.get("social_finance_trend", "—")},

    {"维度": "资金", "核心指标": "M2 同比", "数据源": "中国人民银行",
     "积极区间": ">8% (且M1-M2剪刀差收窄)", "消极区间": "<7%", "信号解读": "广义货币供应，过高可能无效空转，过低则通缩压力大。",
     "key": "m2_yoy", "fmt": lambda v: f"{v:.2f}%",
     "diag_type": "threshold", "hot": 8, "cold": 7, "hot_label": "宽松", "cold_label": "偏紧", "hot_is_high": True},

    {"维度": "资金", "核心指标": "Shibor (隔夜)", "数据源": "银行间市场",
     "积极区间": "利率处于低位", "消极区间": "利率飙升", "信号解读": "银行间资金成本，直接反映市场短期钱紧不紧。",
     "key": "shibor_overnight", "fmt": lambda v: f"{v:.4f}%",
     "diag_type": "percentile", "pct_key": "shibor_pct", "hot_label": "偏紧", "cold_label": "宽松"},
    # 注：Shibor 的 percentile 用 invert 语义不一致，这里直接用低=宽松，分位高=偏紧

    {"维度": "资金", "核心指标": "LPR (1年期)", "数据源": "中国人民银行",
     "积极区间": "下调或维持低位", "消极区间": "上调", "信号解读": "实体贷款利率基准，下调利好企业融资与楼市。",
     "key": "lpr_1y", "fmt": lambda v: f"{v:.2f}%",
     "diag_type": "custom", "fn": lambda v, d: d.get("lpr_trend", "—")},

    {"维度": "资金", "核心指标": "10Y国债收益率", "数据源": "中债登",
     "积极区间": "低位 / 下行", "消极区间": "飙升", "信号解读": "长期利率锚，反映资本环境宽松度。",
     "key": "yield_10y", "fmt": lambda v: f"{v:.2f}%",
     "diag_type": "percentile", "pct_key": "yield_pct", "hot_label": "宽松", "cold_label": "偏紧"},

    # 走势
    {"维度": "走势", "核心指标": "上证指数 (vs 30日线)", "数据源": "全市场收盘价替代",
     "积极区间": "站上均线", "消极区间": "跌破均线", "信号解读": "中短期趋势生命线，线上持股，线下持币。",
     "key": "index_close", "fmt": lambda v: f"{v:.2f}",
     "diag_type": "custom", "fn": lambda v, d: "站上均线" if d.get("index_above_ma30") else "跌破均线"},

    {"维度": "走势", "核心指标": "全A PE (TTM中位数)", "数据源": "akshare",
     "积极区间": "<15倍 (低估)", "消极区间": ">20倍 (泡沫)", "信号解读": "衡量市场贵贱。需结合盈利增速(PEG)观看。",
     "key": "pe", "fmt": lambda v: f"{v:.2f}",
     "diag_type": "threshold", "hot": 20, "cold": 15, "hot_label": "泡沫", "cold_label": "低估", "hot_is_high": True},

    {"维度": "走势", "核心指标": "百城房价指数同比", "数据源": "统计局/中指院",
     "积极区间": "量价齐升 / 企稳", "消极区间": "缩量下跌", "信号解读": "房地产不仅是资产，也是信用的锚，下跌会压制风险偏好。",
     "key": "house_price_yoy", "fmt": lambda v: f"{v:.2f}%",
     "diag_type": "threshold", "hot": 0, "cold": -1, "hot_label": "企稳回升", "cold_label": "下跌", "hot_is_high": True},

    {"维度": "走势", "核心指标": "融资融券余额", "数据源": "交易所",
     "积极区间": "持续增加", "消极区间": "持续减少", "信号解读": "杠杆资金的态度，反映市场风险偏好最高的资金动向。",
     "key": "margin", "fmt": lambda v: f"{v/1e8:.0f} 亿",
     "diag_type": "percentile", "pct_key": "margin_pct", "hot_label": "亢奋", "cold_label": "低迷"},

    # 情绪
    {"维度": "情绪", "核心指标": "新基金发行份额", "数据源": "基金业协会",
     "积极区间": "显著回暖 / 爆款频出", "消极区间": "冰点期 / 发行失败", "信号解读": "散户入场意愿的直接体现，过热通常是顶部信号。",
     "key": None, "fmt": lambda v: "—", "diag_type": "none"},

    {"维度": "情绪", "核心指标": "北向资金 (20日净流入)", "数据源": "东方财富",
     "积极区间": "连续净流入", "消极区间": "连续净流出", "信号解读": "聪明钱动向，对核心资产影响大。",
     "key": "northbound_20d_net", "fmt": lambda v: f"{v/1e8:.2f} 亿",
     "diag_type": "custom", "fn": lambda v, d: d.get("northboard_trend", "—")},

    {"维度": "情绪", "核心指标": "两市成交额", "数据源": "交易所(国证A指)",
     "积极区间": "放量 (>1万亿)", "消极区间": "缩量 (<6000亿)", "信号解读": "量在价先。无量上涨难持续，地量往往见地价。",
     "key": "market_volume", "fmt": lambda v: f"{v:.0f} 亿" if not np.isnan(v) else "—",
     "diag_type": "custom", "fn": lambda v, d: d.get("market_volume_trend", "—")},

    {"维度": "情绪", "核心指标": "投资者情绪指数", "数据源": "互联网/券商",
     "积极区间": "乐观 (需警惕过热)", "消极区间": "悲观 (可能是机会)", "信号解读": "如股吧热度、开户数等，极端悲观往往是左侧买点。",
     "key": None, "fmt": lambda v: "—", "diag_type": "none"},

    # 风险
    {"维度": "风险", "核心指标": "QVIX (50ETF期权波动率)", "数据源": "中证",
     "积极区间": "低位徘徊", "消极区间": "突然飙升", "信号解读": "中国版恐慌指数。飙升代表市场极度恐慌，避险情绪高涨。",
     "key": "qvix", "fmt": lambda v: f"{v:.2f}",
     "diag_type": "percentile", "pct_key": "qvix_pct", "hot_label": "飙升", "cold_label": "低位"},

    {"维度": "风险", "核心指标": "信用利差", "数据源": "Wind/中债登",
     "积极区间": "利差收窄", "消极区间": "利差走阔", "信号解读": "信用债利率减去国债利率。走阔代表企业违约风险上升。",
     "key": None, "fmt": lambda v: "—", "diag_type": "none"},

    {"维度": "风险", "核心指标": "市场风格 (大小盘/成长价值)", "数据源": "行业指数",
     "积极区间": "进攻型板块领涨", "消极区间": "防御/红利板块领涨", "信号解读": "只有防御板块涨时，说明市场缺乏信心。",
     "key": None, "fmt": lambda v: "—", "diag_type": "none"},

    {"维度": "风险", "核心指标": "人民币汇率 (USD/CNY)", "数据源": "外汇交易中心",
     "积极区间": "稳中有升 / 双向波动", "消极区间": "快速贬值", "信号解读": "汇率贬值往往伴随资金外流压力，压制A股估值。",
     "key": "usd_cny", "fmt": lambda v: f"{v:.4f}",
     "diag_type": "custom", "fn": lambda v, d: d.get("fx_trend", "—")},
]

# 构建展示表
display_rows = []
for ind in indicators:
    cur_val, diag = _threshold_diag(ind)
    display_rows.append({
        "维度": ind["维度"],
        "核心指标": ind["核心指标"],
        "数据源": ind["数据源"],
        "✅ 积极区间": ind["积极区间"],
        "⚠️ 消极区间": ind["消极区间"],
        "信号解读": ind["信号解读"],
        "当前值": cur_val,
        "诊断": diag,
    })

display_df = pd.DataFrame(display_rows)

# 高亮诊断列
def _highlight_diag(val):
    if val in ["偏热", "亢奋", "泡沫", "飙升", "贬值", "下行", "跌破均线", "收缩", "疲软", "下跌", "偏紧", "净流出"]:
        return "background-color: #ffe6e6; color: #8B0000"
    if val in ["偏冷", "低迷", "低估", "低位", "升值", "回升", "站上均线", "扩张", "活跃", "旺盛", "企稳回升", "宽松", "净流入"]:
        return "background-color: #e6ffe6; color: #006400"
    if val in ["中性", "持平"]:
        return "background-color: #fff9e6; color: #8B7500"
    return ""

try:
    styled = display_df.style.map(_highlight_diag, subset=["诊断"])
except AttributeError:
    styled = display_df.style.applymap(_highlight_diag, subset=["诊断"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=600)
st.caption("实时拉取的指标已自动诊断（颜色：红=偏热/风险，绿=偏冷/积极，黄=中性）；未实时拉取的指标当前值显示为 —，可作为人工参考。")

# ---------------- 数据源告警 ----------------
if data["errors"]:
    with st.expander(f"⚠️ {len(data['errors'])} 项数据获取失败", expanded=False):
        for msg in data["errors"]:
            st.write(f"- {msg}")
        st.caption("失败项已从综合热度中剔除（按剩余项加权归一）。")

st.divider()

# ---------------- 行为指南 ----------------
st.subheader("💡 周期行为指南")
if heat_score > 70:
    st.error("🚨 当前市场：人群拥挤、激进、高价追涨")
    st.info("**建议风格：审慎且自律、精挑细选**")
elif heat_score < 30:
    st.success("✅ 当前市场：乏人问津、悲观、无心买进")
    st.info("**建议风格：激进、四处投资**")
else:
    st.warning("⚖️ 当前市场：处于中间状态，建议持仓观察。")

st.caption(
    f"综合热度 = PE分位×0.4 + 融资余额分位×0.3 + 宽松度分位×0.3 = {heat_score:.1f}"
)
