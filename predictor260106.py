import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components

# Streamlit 显示 SHAP HTML 图的函数
def st_shap(plot, height=300):
    components.html(plot.html(), height=height)

# ==================== session_state 初始化 ====================
for key in ["prediction_made", "predicted_class", "predicted_proba", "advice", "shap_plot_generated", "feature_values", "features"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.prediction_made = st.session_state.prediction_made or False
st.session_state.shap_plot_generated = st.session_state.shap_plot_generated or False

# ==================== 模型与测试集加载 ====================
model = joblib.load('RF.pkl')
X_test = pd.read_csv('X_test.csv')

# ==================== 特征名称 ====================
feature_names = ["X1","X10","X11","X18","X29","X31","X33"]

# ==================== Streamlit 页面 ====================
st.title("CRKP 预测器")

with st.form("prediction_form"):
    st.subheader("请输入患者信息")
    X1 = st.number_input("X1:", min_value=-10, max_value=10, value=0)
    X10 = st.number_input("X10:", min_value=-10, max_value=10, value=0)
    X11 = st.number_input("X11:", min_value=-10, max_value=10, value=0)
    X18 = st.selectbox("X18:", options=[0,1], format_func=lambda x: "是" if x==1 else "否")
    X29 = st.selectbox("X29:", options=[0,1], format_func=lambda x: "是" if x==1 else "否")
    X31 = st.selectbox("X31:", options=[0,1], format_func=lambda x: "是" if x==1 else "否")
    X33 = st.selectbox("X33:", options=[0,1], format_func=lambda x: "是" if x==1 else "否")
    submitted = st.form_submit_button("Predict")

# ==================== 预测逻辑 ====================
if submitted:
    feature_values = [X1,X10,X11,X18,X29,X31,X33]
    features = pd.DataFrame([feature_values], columns=feature_names)  # ✅ DataFrame 避免 warning

    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 保存 session_state
    st.session_state.feature_values = feature_values
    st.session_state.features = features
    st.session_state.prediction_made = True
    st.session_state.predicted_class = predicted_class
    st.session_state.predicted_proba = predicted_proba
    st.session_state.shap_plot_generated = False

    probability = predicted_proba[1]*100
    if predicted_class == 1:
        advice = f"新生儿患早发型败血症风险较高，概率 {probability:.1f}%。建议立即咨询医生。"
    else:
        advice = f"新生儿患早发型败血症风险较低，概率 {probability:.1f}%。仍需观察，如有异常请就医。"
    st.session_state.advice = advice
    st.success("预测完成！")

# ==================== 显示预测结果 ====================
if st.session_state.prediction_made:
    st.subheader("预测结果")
    class_label = "患病" if st.session_state.predicted_class==1 else "未患病"
    st.write(f"**类别:** {class_label}")
    st.write(f"**患病概率:** {st.session_state.predicted_proba[1]*100:.2f}%")
    st.write(st.session_state.advice)

# ==================== SHAP 解释 ====================
st.subheader("SHAP 力解释图")
if st.session_state.feature_values is not None and not st.session_state.shap_plot_generated:
    
    
    explainer = shap.TreeExplainer(model)
    X_input = pd.DataFrame([st.session_state.feature_values], columns=feature_names)
    shap_values = explainer.shap_values(X_input)
    expected_value = explainer.expected_value
    
    # 二分类模型
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1][0]  # ✅ 这里必须 [0] 取第一条样本
        base_value = expected_value[1]
    else:
        shap_vals_to_plot = shap_values[0]
        base_value = expected_value
    
    shap_html = shap.plots.force(
        base_value,
        shap_vals_to_plot,
        feature_names=feature_names,
        matplotlib=False
    )
    st_shap(shap_html)

    st.session_state.shap_plot_generated = True
elif st.session_state.feature_values is None:
    st.info("请先点击 Predict 查看 SHAP 解释")

# ==================== 清除按钮 ====================
if st.button("清除预测结果"):
    for key in ["prediction_made","predicted_class","predicted_proba","advice","shap_plot_generated","feature_values","features"]:
        st.session_state[key] = None
    st.experimental_rerun()

