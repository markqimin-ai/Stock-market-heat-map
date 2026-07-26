#!/usr/bin/env python3
"""每周一自动生成A股市场评估静态HTML报告。

用法：
    python3 generate_report.py [输出目录]

输出：
    <输出目录>/reports/market_report_YYYYMMDD.html
    <输出目录>/reports/latest.html  （始终指向最新一份）
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html


# ============================================================
#  工具函数
# ============================================================
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
    s = pd.to_numeric(series, errors="coerce").dropna()
    if window_days is not None and dates is not None:
        cutoff = pd.to_datetime(dates).max() - pd.Timedelta(days=window_days)
        mask = pd.to_datetime(dates) >= cutoff
        s = s[mask]
    if len(s) == 0 or pd.isna(current):
        return float("nan")
    rank = (s <= current).mean() * 100.0
    return 100.0 - rank if invert else rank


def _diag_threshold(value, hot_threshold, cold_threshold, hot_label, cold_label,
                    mid_label="中性", hot_is_high=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
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


def _diag_pct(pct, hot_label="偏热", cold_label="偏冷", mid_label="中性"):
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "—"
    if pct >= 70:
        return hot_label
    if pct < 30:
        return cold_label
    return mid_label


def _fmt(v, fmt_str=".2f", unit=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:{fmt_str}}{unit}"


# ============================================================
#  数据抓取
# ============================================================
def fetch_market_data():
    data = {"errors": []}

    def _try(name, fn):
        try:
            return fn()
        except Exception as e:
            data["errors"].append(f"{name}: {e}")
            return None

    # ---- 走势/价格 ----
    pe_df = _try("全市场PE", lambda: ak.stock_a_ttm_lyr())
    if pe_df is not None:
        pe_df["date"] = pd.to_datetime(pe_df["date"])
        pe_df = pe_df.sort_values("date")
        latest = pe_df.iloc[-1]
        data["pe"] = float(latest["middlePETTM"])
        data["pe_pct"] = float(latest["quantileInRecent10YearsMiddlePeTtm"]) * 100
        data["pe_date"] = latest["date"].strftime("%Y-%m-%d")
        data["index_close"] = float(latest["close"])
        close_series = pd.to_numeric(pe_df["close"], errors="coerce")
        data["index_ma30"] = float(close_series.tail(30).mean())
        data["index_above_ma30"] = data["index_close"] > data["index_ma30"]
        # PE历史序列（用于趋势图）
        data["pe_series"] = pe_df[["date", "middlePETTM"]].tail(365)

    # 融资融券
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
        data["margin_series"] = margin_df[[date_col, col]].tail(180).rename(
            columns={date_col: "date", col: "value"})

    # ---- 资金 ----
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

    m2_df = _try("M2", lambda: ak.macro_china_money_supply())
    if m2_df is not None:
        m2_col = _pick_col(m2_df, ["货币和准货币(M2)-同比增长"])
        m1_col = _pick_col(m2_df, ["货币(M1)-同比增长"])
        date_col = _pick_col(m2_df, ["月份", "date", "日期"])
        m2_df = m2_df.sort_values(date_col)
        data["m2_yoy"] = float(pd.to_numeric(m2_df[m2_col], errors="coerce").iloc[-1])
        data["m1_yoy"] = float(pd.to_numeric(m2_df[m1_col], errors="coerce").iloc[-1])
        data["money_date"] = str(m2_df[date_col].iloc[-1])

    sf_df = _try("社融", lambda: ak.macro_china_shrzgm())
    if sf_df is not None:
        col = _pick_col(sf_df, ["社会融资规模增量"])
        date_col = _pick_col(sf_df, ["月份", "date", "日期"])
        sf_df = sf_df.sort_values(date_col)
        series = pd.to_numeric(sf_df[col], errors="coerce")
        data["social_finance"] = float(series.iloc[-1])
        recent3 = series.tail(3).mean()
        prev3 = series.tail(6).head(3).mean()
        data["social_finance_trend"] = "回升" if recent3 > prev3 else "下行"
        data["social_finance_date"] = str(sf_df[date_col].iloc[-1])

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

    lpr_df = _try("LPR", lambda: ak.macro_china_lpr())
    if lpr_df is not None:
        data["lpr_1y"] = float(lpr_df["LPR1Y"].iloc[-1])
        data["lpr_5y"] = float(lpr_df["LPR5Y"].iloc[-1])
        if len(lpr_df) >= 2:
            delta = lpr_df["LPR1Y"].iloc[-1] - lpr_df["LPR1Y"].iloc[-2]
            data["lpr_trend"] = "下调" if delta < 0 else ("上调" if delta > 0 else "持平")
        else:
            data["lpr_trend"] = "—"
        data["lpr_date"] = str(lpr_df["TRADE_DATE"].iloc[-1])

    # ---- 宏观 ----
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
        raw = ip_df[col].iloc[-1]
        if isinstance(raw, str):
            data["industrial_yoy"] = float(raw.replace("%", "").strip())
        else:
            data["industrial_yoy"] = float(pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0])
        data["industrial_date"] = str(ip_df[date_col].iloc[-1])

    retail_df = _try("社消零售", lambda: ak.macro_china_consumer_goods_retail())
    if retail_df is not None:
        col = _pick_col(retail_df, ["同比增长"])
        date_col = _pick_col(retail_df, ["月份", "date", "日期"])
        retail_df = retail_df.sort_values(date_col)
        data["retail_yoy"] = float(pd.to_numeric(retail_df[col], errors="coerce").iloc[-1])
        data["retail_date"] = str(retail_df[date_col].iloc[-1])

    # ---- 情绪 ----
    hsgt_df = _try("北向资金", lambda: ak.stock_hsgt_hist_em(symbol="北向资金"))
    if hsgt_df is not None:
        col = _pick_col(hsgt_df, ["当日成交净买额"])
        date_col = _pick_col(hsgt_df, ["日期", "date"])
        hsgt_df[date_col] = pd.to_datetime(hsgt_df[date_col])
        hsgt_df = hsgt_df.sort_values(date_col)
        series = pd.to_numeric(hsgt_df[col], errors="coerce").dropna()
        if len(series) >= 20:
            data["northbound_20d_net"] = float(series.tail(20).sum())
            data["northboard_trend"] = "净流入" if series.tail(20).sum() > 0 else "净流出"
        data["northboard_date"] = hsgt_df[date_col].iloc[-1].strftime("%Y-%m-%d")

    house_df = _try("百城房价", lambda: ak.macro_china_real_estate())
    if house_df is not None:
        col = _pick_col(house_df, ["涨跌幅"])
        date_col = _pick_col(house_df, ["日期", "date"])
        house_df = house_df.sort_values(date_col)
        data["house_price_yoy"] = float(pd.to_numeric(house_df[col], errors="coerce").iloc[-1])
        data["house_date"] = str(house_df[date_col].iloc[-1])

    # ---- 风险 ----
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

    fx_df = _try("汇率", lambda: ak.currency_boc_safe())
    if fx_df is not None:
        col = _pick_col(fx_df, ["美元"])
        date_col = _pick_col(fx_df, ["日期", "date"])
        fx_df[date_col] = pd.to_datetime(fx_df[date_col])
        fx_df = fx_df.sort_values(date_col)
        series = pd.to_numeric(fx_df[col], errors="coerce").dropna() / 100.0
        data["usd_cny"] = float(series.iloc[-1])
        if len(series) >= 21:
            chg = (series.iloc[-1] / series.iloc[-21] - 1) * 100
            data["fx_20d_chg"] = float(chg)
            data["fx_trend"] = "贬值" if chg > 0 else "升值"
        data["fx_date"] = fx_df[date_col].iloc[-1].strftime("%Y-%m-%d")

    return data


# ============================================================
#  生成 Plotly 图表 HTML
# ============================================================
def make_gauge_fig(value, title, reference=None):
    """生成单温度计图表。"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1a1a2e", "thickness": 0.25},
            "bgcolor": "#f5f5f5",
            "steps": [
                {"range": [0, 30], "color": "#90EE90"},
                {"range": [30, 70], "color": "#F0E68C"},
                {"range": [70, 100], "color": "#DC143C"},
            ],
            "threshold": {
                "line": {"color": "#8B0000", "width": 3},
                "thickness": 0.75, "value": 90,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
    )
    return fig


def gauge_to_html(fig):
    return to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


# ============================================================
#  生成完整 HTML
# ============================================================
def build_report(data, heat_score, report_date):
    """构建完整HTML报告字符串。"""

    indicators = [
        # (维度, 指标, 数据源, 积极区间, 消极区间, 解读, 当前值, 诊断, 诊断类别)
        ("宏观", "GDP 同比", "国家统计局", "&gt;5%", "&lt;3%",
         "衡量经济总量增长动能",
         _fmt(data.get("gdp_yoy"), ".2f", "%"),
         _diag_threshold(data.get("gdp_yoy"), 5, 3, "扩张", "收缩")),

        ("宏观", "PMI (制造业)", "统计局", "&gt;50 (荣枯线上)", "&lt;50 (荣枯线下)",
         "环比动能指标，&gt;50代表扩张",
         _fmt(data.get("pmi_mfg"), ".2f"),
         _diag_threshold(data.get("pmi_mfg"), 50, 50, "扩张", "收缩")),

        ("宏观", "PMI (非制造业)", "统计局", "&gt;50", "&lt;50",
         "非制造业（服务业/建筑业）景气度",
         _fmt(data.get("pmi_nonmfg"), ".2f"),
         _diag_threshold(data.get("pmi_nonmfg"), 50, 50, "扩张", "收缩")),

        ("宏观", "工业增加值同比", "国家统计局", "&gt;6%", "&lt;4%",
         "反映生产端活跃度",
         _fmt(data.get("industrial_yoy"), ".2f", "%"),
         _diag_threshold(data.get("industrial_yoy"), 6, 4, "活跃", "疲软")),

        ("宏观", "社消零售总额同比", "国家统计局", "&gt;6%", "&lt;4%",
         "反映内需消费能力",
         _fmt(data.get("retail_yoy"), ".2f", "%"),
         _diag_threshold(data.get("retail_yoy"), 6, 4, "旺盛", "疲软")),

        ("资金", "社融规模增量", "人行/统计局", "企稳回升", "持续下行",
         "实体经济融资需求",
         _fmt(data.get("social_finance"), ".0f", " 亿"),
         data.get("social_finance_trend", "—")),

        ("资金", "M2 同比", "中国人民银行", "&gt;8%", "&lt;7%",
         "广义货币供应",
         _fmt(data.get("m2_yoy"), ".2f", "%"),
         _diag_threshold(data.get("m2_yoy"), 8, 7, "宽松", "偏紧")),

        ("资金", "Shibor (隔夜)", "银行间市场", "利率低位", "利率飙升",
         "银行间资金成本",
         _fmt(data.get("shibor_overnight"), ".4f", "%"),
         _diag_pct(data.get("shibor_pct"), "偏紧", "宽松")),

        ("资金", "LPR (1年期)", "中国人民银行", "下调或低位", "上调",
         "实体贷款利率基准",
         _fmt(data.get("lpr_1y"), ".2f", "%"),
         data.get("lpr_trend", "—")),

        ("资金", "10Y国债收益率", "中债登", "低位 / 下行", "飙升",
         "长期利率锚，反映资本环境宽松度",
         _fmt(data.get("yield_10y"), ".2f", "%"),
         _diag_pct(data.get("yield_pct"), "宽松", "偏紧")),

        ("走势", "全市场收盘价 (vs 30日线)", "akshare", "站上均线", "跌破均线",
         "中短期趋势生命线",
         _fmt(data.get("index_close"), ".2f"),
         "站上均线" if data.get("index_above_ma30") else "跌破均线"),

        ("走势", "全A PE (TTM中位数)", "akshare", "&lt;15倍 (低估)", "&gt;20倍 (泡沫)",
         "衡量市场贵贱",
         _fmt(data.get("pe"), ".2f"),
         _diag_threshold(data.get("pe"), 20, 15, "泡沫", "低估")),

        ("走势", "百城房价指数同比", "统计局/中指院", "企稳回升", "缩量下跌",
         "房地产是信用的锚，下跌压制风险偏好",
         _fmt(data.get("house_price_yoy"), ".2f", "%"),
         _diag_threshold(data.get("house_price_yoy"), 0, -1, "企稳回升", "下跌")),

        ("走势", "融资融券余额", "交易所", "持续增加", "持续减少",
         "杠杆资金态度，反映风险偏好",
         _fmt(data.get("margin")/1e8 if data.get("margin") else None, ".0f", " 亿")
            if data.get("margin") else "—",
         _diag_pct(data.get("margin_pct"), "亢奋", "低迷")),

        ("情绪", "新基金发行份额", "基金业协会", "回暖 / 爆款", "冰点期",
         "散户入场意愿",
         "—", "—"),

        ("情绪", "北向资金 (20日净流入)", "东方财富", "连续净流入", "连续净流出",
         "聪明钱动向，影响核心资产",
         _fmt(data.get("northbound_20d_net")/1e8
              if data.get("northbound_20d_net") else None, ".2f", " 亿")
            if data.get("northbound_20d_net") else "—",
         data.get("northboard_trend", "—")),

        ("情绪", "两市成交额", "交易所", "放量 (&gt;1万亿)", "缩量 (&lt;6000亿)",
         "量在价先",
         "—", "—"),

        ("情绪", "投资者情绪指数", "互联网/券商", "乐观 (警惕过热)", "悲观 (可能是机会)",
         "极端悲观是左侧买点",
         "—", "—"),

        ("风险", "QVIX (50ETF波动率)", "中证", "低位徘徊", "突然飙升",
         "中国版恐慌指数",
         _fmt(data.get("qvix"), ".2f"),
         _diag_pct(data.get("qvix_pct"), "飙升", "低位")),

        ("风险", "信用利差", "Wind/中债登", "利差收窄", "利差走阔",
         "企业违约风险",
         "—", "—"),

        ("风险", "市场风格", "行业指数", "进攻型领涨", "防御型领涨",
         "防御独涨 = 缺信心",
         "—", "—"),

        ("风险", "人民币汇率 (USD/CNY)", "外汇交易中心", "稳中有升", "快速贬值",
         "贬值压制A股估值",
         _fmt(data.get("usd_cny"), ".4f"),
         data.get("fx_trend", "—")),
    ]

    # 诊断类别（用于上色）
    hot_set = {"偏热", "亢奋", "泡沫", "飙升", "贬值", "下行", "跌破均线",
               "收缩", "疲软", "下跌", "偏紧", "净流出"}
    cold_set = {"偏冷", "低迷", "低估", "低位", "升值", "回升", "站上均线",
                "扩张", "活跃", "旺盛", "企稳回升", "宽松", "净流入"}
    mid_set = {"中性", "持平"}

    def diag_class(d):
        if d in hot_set:
            return "diag-hot"
        if d in cold_set:
            return "diag-cold"
        if d in mid_set:
            return "diag-mid"
        return ""

    # 构建表格行 HTML
    dim_colors = {
        "宏观": "#e8f4f8", "资金": "#fff4e8", "走势": "#e8f8e8",
        "情绪": "#f8e8f8", "风险": "#f8e8e8"
    }
    dim_current = None
    table_rows = []
    for dim, ind, src, pos, neg, interp, val, diag in indicators:
        dc = diag_class(diag)
        if dim != dim_current:
            dcolor = dim_colors.get(dim, "#f5f5f5")
            dim_row = f'<tr class="dim-row" style="background:{dcolor}"><td colspan="6" class="dim-label">{dim}</td></tr>'
            table_rows.append(dim_row)
            dim_current = dim
        table_rows.append(
            f'<tr>'
            f'<td class="col-ind">{ind}</td>'
            f'<td class="col-src">{src}</td>'
            f'<td class="col-pos">{pos}</td>'
            f'<td class="col-neg">{neg}</td>'
            f'<td class="col-val">{val}</td>'
            f'<td class="col-diag {dc}">{diag}</td>'
            f'</tr>'
        )
    table_html = "\n".join(table_rows)

    # 主温度计 HTML
    main_gauge = gauge_to_html(make_gauge_fig(heat_score, "市场综合热度 (0-100)"))

    # 三分项温度计
    sub_gauges_html = ""
    sub_items = [
        (data.get("pe_pct"), "资产价格 (PE)", "近10年分位"),
        (data.get("margin_pct"), "投资人情绪 (融资)", "近5年分位"),
        (data.get("yield_pct"), "资本环境 (10Y国债)", "宽松度分位"),
    ]
    for val, title, sub in sub_items:
        if val is None or np.isnan(val):
            continue
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(val, 1),
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1a1a2e", "thickness": 0.25},
                "bgcolor": "#f5f5f5",
                "steps": [
                    {"range": [0, 30], "color": "#90EE90"},
                    {"range": [30, 70], "color": "#F0E68C"},
                    {"range": [70, 100], "color": "#DC143C"},
                ],
            },
        ))
        fig.update_layout(height=230, margin=dict(l=5, r=5, t=50, b=5), paper_bgcolor="white")
        sub_gauges_html += f'<div class="sub-gauge">{to_html(fig, include_plotlyjs=False, full_html=False)}</div>'

    # 行为建议
    if heat_score > 70:
        advice_title = "🚨 当前市场：人群拥挤、激进、高价追涨"
        advice_style = "advice-hot"
        advice_body = "<strong>建议风格：审慎且自律、精挑细选</strong><br>" \
                      "• 减少新增仓位，优先止盈高估标的<br>" \
                      "• 提高现金/债券等防御性资产比例<br>" \
                      "• 只买深度研究、有安全边际的个股"
    elif heat_score < 30:
        advice_title = "✅ 当前市场：乏人问津、悲观、无心买进"
        advice_style = "advice-cold"
        advice_body = "<strong>建议风格：激进、四处投资</strong><br>" \
                      "• 逐步加仓优质标的，越跌越买<br>" \
                      "• 关注被错杀的成长股和高股息龙头<br>" \
                      "• 保持耐心，左侧布局，等待周期反转"
    else:
        advice_title = "⚖️ 当前市场：处于中间状态，建议持仓观察"
        advice_style = "advice-mid"
        advice_body = "<strong>建议风格：平衡持仓，等待信号</strong><br>" \
                      "• 维持中性仓位，不追涨不杀跌<br>" \
                      "• 关注资金面与基本面的背离能否修复<br>" \
                      "• 结构上偏向防御 + 成长的均衡配置"

    # 失败项
    errors_html = ""
    if data["errors"]:
        errors_html = '<div class="errors"><strong>⚠️ 数据获取失败项：</strong><ul>' + \
                      "".join(f"<li>{e}</li>" for e in data["errors"]) + "</ul></div>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>A股市场温度计 - {report_date}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Helvetica Neue",
                 "Microsoft YaHei", sans-serif;
    background: #f0f2f5;
    color: #1a1a2e;
    line-height: 1.6;
    padding-bottom: 30px;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white;
    padding: 20px 16px 16px;
    text-align: center;
  }}
  .header h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 4px; }}
  .header .sub {{ font-size: 12px; opacity: 0.8; }}
  .header .date {{ font-size: 11px; opacity: 0.6; margin-top: 6px; }}

  .container {{ padding: 12px; max-width: 768px; margin: 0 auto; }}

  .card {{
    background: white;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  .card h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 10px; color: #1a1a2e; }}

  .main-gauge {{ text-align: center; }}
  .main-gauge .plotly {{ width: 100% !important; }}

  .sub-gauges {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
  }}
  .sub-gauge {{ background: #fafafa; border-radius: 8px; padding: 4px; }}
  .sub-gauge .plotly {{ width: 100% !important; }}

  .advice {{ border-radius: 10px; padding: 14px; }}
  .advice-hot {{ background: #ffe6e6; border-left: 4px solid #8B0000; }}
  .advice-cold {{ background: #e6ffe6; border-left: 4px solid #006400; }}
  .advice-mid {{ background: #fff9e6; border-left: 4px solid #8B7500; }}
  .advice-title {{ font-size: 14px; font-weight: 600; margin-bottom: 6px; }}
  .advice-body {{ font-size: 13px; line-height: 1.8; }}

  .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{
    background: #1a1a2e; color: white; font-weight: 500;
    padding: 8px 6px; text-align: left; position: sticky; top: 0; z-index: 1;
    font-size: 11px;
  }}
  td {{ padding: 7px 6px; border-bottom: 1px solid #f0f0f0; }}
  .dim-row td {{
    font-weight: 600; font-size: 12px;
    padding: 6px 10px; border-bottom: 1px solid #ddd;
  }}
  .col-ind {{ font-weight: 500; min-width: 110px; }}
  .col-src {{ color: #888; min-width: 80px; font-size: 11px; }}
  .col-pos {{ color: #006400; min-width: 100px; font-size: 11px; }}
  .col-neg {{ color: #8B0000; min-width: 100px; font-size: 11px; }}
  .col-val {{ font-weight: 500; min-width: 80px; text-align: right; font-variant-numeric: tabular-nums; }}
  .col-diag {{ min-width: 70px; text-align: center; font-weight: 600; font-size: 11px; }}
  .diag-hot {{ color: #8B0000; background: #ffe6e6; border-radius: 4px; padding: 2px 6px; display: inline-block; }}
  .diag-cold {{ color: #006400; background: #e6ffe6; border-radius: 4px; padding: 2px 6px; display: inline-block; }}
  .diag-mid {{ color: #8B7500; background: #fff9e6; border-radius: 4px; padding: 2px 6px; display: inline-block; }}

  .errors {{
    background: #fff4f4; border-radius: 8px; padding: 10px 12px;
    font-size: 12px; color: #8B0000; margin-top: 8px;
  }}
  .errors ul {{ margin-left: 18px; margin-top: 4px; }}

  .footer {{
    text-align: center; font-size: 11px; color: #999;
    padding: 20px 0;
  }}

  .core-stats {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-bottom: 10px;
  }}
  .stat-item {{
    background: #fafafa;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
  }}
  .stat-item .label {{ font-size: 11px; color: #666; margin-bottom: 4px; }}
  .stat-item .value {{ font-size: 16px; font-weight: 600; color: #1a1a2e; font-variant-numeric: tabular-nums; }}
  .stat-item .sub {{ font-size: 10px; color: #999; margin-top: 2px; }}

  @media (max-width: 480px) {{
    .sub-gauges {{ grid-template-columns: 1fr; }}
    .core-stats {{ grid-template-columns: 1fr 1fr; }}
    .header h1 {{ font-size: 17px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 A股市场评估工具</h1>
  <div class="sub">基于霍华德·马克斯周期理论</div>
  <div class="date">报告日期：{report_date} ｜ 数据来源：akshare</div>
</div>

<div class="container">

  <!-- 主温度计 -->
  <div class="card main-gauge">
    {main_gauge}
  </div>

  <!-- 行为建议 -->
  <div class="card advice {advice_style}">
    <div class="advice-title">{advice_title}</div>
    <div class="advice-body">{advice_body}</div>
  </div>

  <!-- 三分项温度计 -->
  <div class="card">
    <h2>🔥 三大维度热度</h2>
    <div class="sub-gauges">
      {sub_gauges_html}
    </div>
  </div>

  <!-- 核心指标速览 -->
  <div class="card">
    <h2>📈 核心指标速览</h2>
    <div class="core-stats">
      <div class="stat-item">
        <div class="label">全市场 PE</div>
        <div class="value">{_fmt(data.get("pe"), ".2f")}</div>
        <div class="sub">分位 {_fmt(data.get("pe_pct"), ".1f", "%")}</div>
      </div>
      <div class="stat-item">
        <div class="label">融资余额</div>
        <div class="value">{_fmt(data.get("margin")/1e8 if data.get("margin") else None, ".0f", "亿")
                            if data.get("margin") else "—"}</div>
        <div class="sub">分位 {_fmt(data.get("margin_pct"), ".1f", "%")}</div>
      </div>
      <div class="stat-item">
        <div class="label">10Y 国债</div>
        <div class="value">{_fmt(data.get("yield_10y"), ".2f", "%")}</div>
        <div class="sub">宽松度 {_fmt(data.get("yield_pct"), ".1f", "%")}</div>
      </div>
      <div class="stat-item">
        <div class="label">PMI 制造业</div>
        <div class="value">{_fmt(data.get("pmi_mfg"), ".1f")}</div>
        <div class="sub">{_diag_threshold(data.get("pmi_mfg"), 50, 50, "扩张", "收缩")}</div>
      </div>
      <div class="stat-item">
        <div class="label">M2 同比</div>
        <div class="value">{_fmt(data.get("m2_yoy"), ".1f", "%")}</div>
        <div class="sub">{_diag_threshold(data.get("m2_yoy"), 8, 7, "宽松", "偏紧")}</div>
      </div>
      <div class="stat-item">
        <div class="label">北向 20日</div>
        <div class="value">{_fmt(data.get("northbound_20d_net")/1e8
                                 if data.get("northbound_20d_net") else None, ".1f", "亿")
                            if data.get("northbound_20d_net") else "—"}</div>
        <div class="sub">{data.get("northboard_trend", "—")}</div>
      </div>
    </div>
  </div>

  <!-- 完整指标体系 -->
  <div class="card">
    <h2>📚 完整指标体系 (20项)</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>核心指标</th>
            <th>数据源</th>
            <th>✅ 积极</th>
            <th>⚠️ 消极</th>
            <th>当前值</th>
            <th>诊断</th>
          </tr>
        </thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </div>
    {errors_html}
  </div>

  <div class="footer">
    综合热度 = PE分位×0.4 + 融资分位×0.3 + 宽松度分位×0.3<br>
    本报告仅供参考，不构成投资建议
  </div>

</div>
</body>
</html>"""
    return html


# ============================================================
#  主入口
# ============================================================
def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("正在拉取市场数据…")
    data = fetch_market_data()
    print(f"数据拉取完成，失败 {len(data['errors'])} 项")

    # 计算综合热度
    parts, weights = [], []
    for key, w in [("pe_pct", 0.4), ("margin_pct", 0.3), ("yield_pct", 0.3)]:
        v = data.get(key)
        if v is not None and not np.isnan(v):
            parts.append(v * w)
            weights.append(w)
    heat_score = sum(parts) / sum(weights) if weights else 50.0
    heat_score = float(np.clip(heat_score, 0, 100))
    print(f"综合热度: {heat_score:.1f}")

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_stamp = datetime.now().strftime("%Y%m%d")

    # 生成 HTML
    html = build_report(data, heat_score, report_date)

    # 保存
    dated_file = reports_dir / f"market_report_{date_stamp}.html"
    latest_file = reports_dir / "latest.html"

    dated_file.write_text(html, encoding="utf-8")
    latest_file.write_text(html, encoding="utf-8")

    print(f"报告已保存: {dated_file}")
    print(f"最新报告: {latest_file}")
    return latest_file


if __name__ == "__main__":
    main()
