import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid

# 🔹 初始化 session_state
for key in ["prediction_made", "predicted_class", "predicted_proba", "advice",
            "shap_plot_generated", "feature_values", "features"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.shap_plot_generated = st.session_state.shap_plot_generated or False

# 🔹 加载模型和测试集
model = joblib.load("RF.pkl")
X_test = pd.read_csv("X_test.csv")

# 🔹 特征名称
feature_names = ["X1", "X10", "X11", "X18", "X29", "X31", "X33"]

# 🔹 Streamlit 页面
st.title("CRKP预测器")

# ===== 输入表单 =====
with st.form("prediction_form"):
    st.subheader("请输入患者信息")
    X1 = st.number_input("X1:", min_value=-10, max_value=10, value=0)
    X10 = st.number_input("X10:", min_value=-10, max_value=10, value=0)
    X11 = st.number_input("白细胞:", min_value=-10, max_value=10, value=0)
    X18 = st.selectbox("X18:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    X29 = st.selectbox("发热:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    X31 = st.selectbox("鼻塞:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    X33 = st.selectbox("流产:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    submitted = st.form_submit_button("Predict")

# ===== 预测逻辑 =====
if submitted:
    feature_values = [X1, X10, X11, X18, X29, X31, X33]
    features = np.array([feature_values])

    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    st.session_state.prediction_made = True
    st.session_state.predicted_class = predicted_class
    st.session_state.predicted_proba = predicted_proba
    st.session_state.feature_values = feature_values
    st.session_state.features = features
    st.session_state.shap_plot_generated = False

    probability = predicted_proba[1] * 100
    if predicted_class == 1:
        advice = (
            f"根据我们的模型，新生儿患有早发型败血症的风险较高。 "
            f"模型预测患病概率为 {probability:.1f}%。 "
            "建议立即咨询医疗保健提供者进行进一步评估和可能的干预。"
        )
    else:
        advice = (
            f"根据我们的模型，新生儿患有早发型败血症的风险较低。 "
            f"模型预测患病概率为 {probability:.1f}%。 "
            "仍需密切观察新生儿状况，如有异常请及时就医。"
        )
    st.session_state.advice = advice
    st.success("预测完成！")

# ===== 显示预测结果 =====
if st.session_state.prediction_made:
    st.subheader("预测结果：")
    class_label = "患病" if st.session_state.predicted_class == 1 else "未患病 (0)"
    st.write(f"**预测类别:** {class_label}")
    st.write(f"**患病概率:** {st.session_state.predicted_proba[1]*100:.2f}%")
    st.write(st.session_state.advice)

    # ===== SHAP 解释 =====
    st.subheader("SHAP 力解释图（始终显示阳性类别）")
    if not st.session_state.shap_plot_generated:
        input_df = pd.DataFrame([st.session_state.feature_values], columns=feature_names)
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(input_df)
        shap_values_pos = shap_values[1]
        expected_value_pos = explainer_shap.expected_value[1]

        # 计算 SHAP 对应概率（与 predict_proba 一致）
        shap_prob = expit(expected_value_pos + shap_values_pos.sum())

        plt.figure(figsize=(10, 6))
        shap.force_plot(
            expected_value=expected_value_pos,
            shap_values=shap_values_pos,
            features=input_df,
            matplotlib=True,
            show=False
        )
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.session_state.shap_plot_generated = True

    st.image("shap_force_plot.png", caption=f"SHAP 力解释图（阳性类别） - 预测概率约 {shap_prob:.2f}")

    # ===== 清除预测结果 =====
    if st.button("清除预测结果", type="primary"):
        for key in ["prediction_made", "predicted_class", "predicted_proba",
                    "advice", "shap_plot_generated", "feature_values", "features"]:
            st.session_state[key] = None
        st.rerun()
