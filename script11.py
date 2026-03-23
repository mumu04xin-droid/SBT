# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os

# ==================== 1. 配置与路径 ====================
# 请确保此路径指向您的实际数据文件
DATA_PATH = r"C:\Users\24245\Desktop\data\sbt_monitoring_data.csv"


# ==================== 2. 分段赋分系统 ====================
class WeaningScoreSystem:
    RR_SCORES = {(0, 20): 4, (20, 25): 3, (25, 30): 2, (30, 35): 1, (35, 200): 0}
    HR_SCORES = {(0, 80): 4, (80, 100): 3, (100, 110): 2, (110, 120): 1, (120, 300): 0}
    SPO2_SCORES = {(0, 90): 0, (90, 93): 1, (93, 95): 2, (95, 97): 3, (97, 100): 4}
    PA_SCORES = {(0, 30): 4, (30, 60): 3, (60, 90): 2, (90, 120): 1, (120, 180): 0}
    SBP_SCORES = {(0, 90): 0, (90, 110): 3, (110, 140): 4, (140, 160): 2, (160, 300): 1}
    TV_SCORES = {(0, 350): 0, (350, 450): 2, (450, 550): 3, (550, 700): 4, (700, 2000): 3}

    @classmethod
    def _get_score(cls, value, score_dict):
        for (low, high), score in score_dict.items():
            if low <= value < high: return score
        return 0

    @classmethod
    def calculate_single_score(cls, rr, hr, spo2, pa, sbp, tv):
        scores = {
            'rr_score': cls._get_score(rr, cls.RR_SCORES),
            'hr_score': cls._get_score(hr, cls.HR_SCORES),
            'spo2_score': cls._get_score(spo2, cls.SPO2_SCORES),
            'pa_score': cls._get_score(pa, cls.PA_SCORES),
            'sbp_score': cls._get_score(sbp, cls.SBP_SCORES),
            'tv_score': cls._get_score(tv, cls.TV_SCORES),
        }
        scores['total_score'] = sum(scores.values())
        return scores


# ==================== 3. 数据处理函数 ====================
def get_random_patient_from_db(outcome_target):
    if not os.path.exists(DATA_PATH):
        st.error(f"文件未找到: {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    target_ids = df[df['outcome'] == outcome_target]['patient_id'].unique()
    if len(target_ids) == 0: return None

    sid = np.random.choice(target_ids)
    p_data = df[df['patient_id'] == sid].sort_values('time_point').head(15)

    return pd.DataFrame({
        '时间(min)': [f'T{i * 2}' for i in range(15)],
        '呼吸频率\n(次/分钟)': p_data['respiratory_rate'].values,
        '心率\n(次/分钟)': p_data['heart_rate'].values,
        '血氧饱和度\n(%)': p_data['spo2'].values,
        '相位角\n(度)': p_data['phase_angle'].values,
        '收缩压\n(mmHg)': p_data['systolic_bp'].values,
        '潮气量\n(mL)': p_data['tidal_volume'].values
    }), sid


# ==================== 4. 主界面模式 ====================
def main():
    st.set_page_config(page_title="SBT监测评估系统", layout="wide")

    # 顶部标题栏
    st.markdown("""
        <div style="background-color: #2E86AB; padding: 15px; border-radius: 5px; color: white; text-align: center;">
            <h2 style="margin:0;">基于胸腹运动监测的撤机评估模型</h2>
            <p style="margin:0; font-size:14px;"></p>
        </div>
    """, unsafe_allow_html=True)

    # 数据输入区
    st.markdown("### SBT监测数据输入")
    c1, c2, c3, _ = st.columns([1.2, 1.2, 1, 4])

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame(
            {'时间(min)': [f'T{i * 2}' for i in range(15)], '呼吸频率\n(次/分钟)': [20.0] * 15,
             '心率\n(次/分钟)': [80.0] * 15, '血氧饱和度\n(%)': [98.0] * 15, '相位角\n(度)': [20.0] * 15,
             '收缩压\n(mmHg)': [120.0] * 15, '潮气量\n(mL)': [550.0] * 15})
        st.session_state.pid = "未加载"

    if c1.button("加载撤机成功案例"):
        res = get_random_patient_from_db(1)
        if res: st.session_state.df, st.session_state.pid = res

    if c2.button("加载撤机失败案例"):
        res = get_random_patient_from_db(0)
        if res: st.session_state.df, st.session_state.pid = res

    if c3.button("清空数据"):
        st.session_state.df = pd.DataFrame(
            {'时间(min)': [f'T{i * 2}' for i in range(15)], '呼吸频率\n(次/分钟)': [np.nan] * 15,
             '心率\n(次/分钟)': [np.nan] * 15, '血氧饱和度\n(%)': [np.nan] * 15, '相位角\n(度)': [np.nan] * 15,
             '收缩压\n(mmHg)': [np.nan] * 15, '潮气量\n(mL)': [np.nan] * 15})
        st.session_state.pid = "已清空"

    edited_df = st.data_editor(st.session_state.df, use_container_width=True, hide_index=True)

    if st.button("开始评估", type="primary", use_container_width=True):
        # 1. 计算
        row_scores = [WeaningScoreSystem.calculate_single_score(r[1], r[2], r[3], r[4], r[5], r[6]) for r in
                      edited_df.itertuples(index=False)]
        scores_df = pd.DataFrame(row_scores)
        avg_score = scores_df['total_score'].mean()
        trend = scores_df['total_score'].iloc[-3:].mean() - scores_df['total_score'].iloc[:3].mean()

        # 模拟概率逻辑
        prob = 1 / (1 + np.exp(-(avg_score - 13) * 0.7))
        cat = "高概率成功" if prob >= 0.8 else "中等概率成功" if prob >= 0.6 else "较低概率成功" if prob >= 0.4 else "低概率成功"
        cat_color = "#28A745" if prob >= 0.8 else "#FFC107" if prob >= 0.6 else "#DC3545"

        # 2. 结果展示区 (第一版模式)
        st.markdown("### 评估结果")
        res_c1, res_c2, res_c3 = st.columns([1.5, 1, 2])

        with res_c1:  # 成功率大卡片
            st.markdown(f"""
                <div style="background-color: #f0f8ff; padding: 25px; border-radius: 10px; text-align: center; border: 1px solid #d1e9ff;">
                    <p style="color: #666; margin:0;">撤机成功率</p>
                    <h1 style="color: #2E86AB; font-size: 50px; margin: 10px 0;">{prob * 100:.1f}%</h1>
                    <span style="background-color: {cat_color}; color: white; padding: 5px 15px; border-radius: 15px;">{cat}</span>
                </div>
            """, unsafe_allow_html=True)

        with res_c2:  # 综合评分
            st.markdown(f"""
                <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; height: 185px;">
                    <p style="color: #666; margin:0;">综合评分</p>
                    <p style="font-size: 18px; margin: 15px 0 5px 0;">平均得分 <b style="color:#A23B72; font-size:24px;">{avg_score:.1f}</b> / 24分</p>
                    <p style="color: #666;">评分趋势 <b style="color:{'green' if trend >= 0 else 'red'}">{'↑' if trend >= 0 else '↓'} {abs(trend):.1f}</b></p>
                </div>
            """, unsafe_allow_html=True)

        with res_c3:  # 指标摘要表
            st.markdown(f"""
                <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; height: 185px;">
                    <p style="font-weight: bold; color: #666; margin-bottom:10px;">指标摘要 (Patient ID: {st.session_state.pid})</p>
                    <table style="width:100%; font-size:13px;">
                        <tr><td>平均呼吸率</td><td><b>{edited_df.iloc[:, 1].mean():.1f}</b> 次/min</td><td>平均心率</td><td><b>{edited_df.iloc[:, 2].mean():.1f}</b> 次/min</td></tr>
                        <tr><td>平均血氧</td><td><b>{edited_df.iloc[:, 3].mean():.1f}</b> %</td><td>平均相位角</td><td><b>{edited_df.iloc[:, 4].mean():.1f}</b> °</td></tr>
                        <tr><td>平均收缩压</td><td><b>{edited_df.iloc[:, 5].mean():.1f}</b> mmHg</td><td>平均潮气量</td><td><b>{edited_df.iloc[:, 6].mean():.0f}</b> mL</td></tr>
                    </table>
                </div>
            """, unsafe_allow_html=True)

        # 3. 各项指标评分 (进度条)
        st.markdown("#### 各项指标评分")
        p_c1, p_c2, p_c3 = st.columns(3)
        metrics = [("呼吸频率", "rr_score", p_c1), ("心率", "hr_score", p_c2), ("血氧饱和度", "spo2_score", p_c3),
                   ("相位角", "pa_score", p_c1), ("收缩压", "sbp_score", p_c2), ("潮气量", "tv_score", p_c3)]
        for label, key, col in metrics:
            val = scores_df[key].mean()
            col.caption(f"{label} {val:.1f} / 4")
            col.progress(val / 4.0)

        # 4. 趋势图
        st.markdown("#### 指标变化趋势")
        plt.figure(figsize=(12, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 标准化以便观察趋势
        for i, col_name in enumerate(edited_df.columns[1:]):
            vals = edited_df[col_name]
            norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-5)
            plt.plot(edited_df['时间(min)'], norm, label=col_name.split('\n')[0], marker='o', markersize=4)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(plt)


if __name__ == "__main__":
    main()