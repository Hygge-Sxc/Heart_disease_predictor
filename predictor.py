import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('RF.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test.csv')# 加载 X_test.csv 文件，主要用作 LIME 解释器的背景数据

# Define feature names from the new dataset
# 手动定义特征名称列表，顺序必须与模型训练时使用的特征顺序完全一致
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

##构建 Streamlit 用户界面 (UI)
# Streamlit user interface
st.title("Heart Disease Predictor")

# Age: numerical input
age = st.number_input("Age:", min_value=0, max_value=120, value=41)# 年龄，默认值41

# Sex: categorical selection
sex = st.selectbox("Sex:", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")# 性别，0代表女性，1代表男性

# Chest Pain Type (cp): categorical selection
cp = st.selectbox("Chest Pain Type (CP):", options=[0, 1, 2, 3])

# Resting Blood Pressure (trestbps): numerical input
trestbps = st.number_input("Resting Blood Pressure (trestbps):", min_value=50, max_value=200, value=120)

# Cholesterol (chol): numerical input
chol = st.number_input("Cholesterol (chol):", min_value=100, max_value=600, value=157)

# Fasting Blood Sugar > 120 mg/dl (fbs): categorical selection
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (FBS):", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Resting Electrocardiographic Results (restecg): categorical selection
restecg = st.selectbox("Resting ECG (restecg):", options=[0, 1, 2])

# Maximum Heart Rate Achieved (thalach): numerical input
thalach = st.number_input("Maximum Heart Rate Achieved (thalach):", min_value=60, max_value=220, value=182)

# Exercise Induced Angina (exang): categorical selection
exang = st.selectbox("Exercise Induced Angina (exang):", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# ST Depression Induced by Exercise (oldpeak): numerical input
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)

# Slope of the Peak Exercise ST Segment (slope): categorical selection
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope):", options=[0, 1, 2])

# Number of Major Vessels Colored by Fluoroscopy (ca): categorical selection
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca):", options=[0, 1, 2, 3, 4])

# Thalassemia (thal): categorical selection
thal = st.selectbox("Thalassemia (thal):", options=[0, 1, 2, 3])

##处理用户输入、进行预测并显示结果 (在 "Predict" 按钮点击后)
# Process inputs and make predictions
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
features = np.array([feature_values])

if st.button("Predict"): # 当用户点击 "Predict" 按钮时，以下代码块执行
    # 使用加载的模型进行预测
    predicted_class = model.predict(features)[0]         # 获取预测的类别 (0 或 1)，[0] 是因为输入只有一个样本
    predicted_proba = model.predict_proba(features)[0]   # 获取属于每个类别的概率，[0] 同上

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}") # 显示例如 [0.9, 0.1] 表示90%概率为0类，10%为1类

    # 根据预测结果生成建议文本
    # probability = predicted_proba[predicted_class] * 100 # 获取预测出类别的概率值
    # 修正：如果predicted_class=0 (No Disease), 我们应该关心其不患病的概率，即predicted_proba[0]
    # 如果predicted_class=1 (Disease), 我们应该关心其患病的概率，即predicted_proba[1]
    if predicted_class == 1: # 预测为有病
        probability_of_disease = predicted_proba[1] * 100
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability_of_disease:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    else: # 预测为无病
        probability_of_no_disease = predicted_proba[0] * 100
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability_of_no_disease:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    st.write(advice) # 显示建议

    ##SHAP 模型解释 (在 "Predict" 按钮点击后)
# SHAP Explanation
    st.subheader("SHAP Force Plot Explanation") # SHAP解释部分的子标题
    explainer_shap = shap.TreeExplainer(model)   # 为加载的树模型创建SHAP解释器

    # 将用户输入的特征值转换为 Pandas DataFrame，并提供特征名称，以便 SHAP 使用
    # 这是因为 TreeExplainer 在处理带特征名的 DataFrame 时效果更好，且输出更易读
    shap_input_df = pd.DataFrame([feature_values], columns=feature_names)
    shap_values = explainer_shap.shap_values(shap_input_df)

    # 显示 SHAP 力图 (force plot)
    # shap_values 对于二分类通常是一个包含两个数组的列表: [shap_values_for_class_0, shap_values_for_class_1]
    # expected_value 也是一个包含两个基准值的列表/数组
    
    # 确保我们使用的是针对单个样本的正确SHAP值和期望值
    # shap_values[class_index] 会得到 (n_samples, n_features) 的数组，这里 n_samples=1
    # 所以 shap_values[class_index][0] 是 (n_features,) 的数组，适合 force_plot
    
    # st.pyplot() 是 Streamlit 中显示 matplotlib 图形的标准方式
    # 我们需要捕获 SHAP 生成的图形对象
    fig, ax = plt.subplots(figsize=(10, 3), dpi=1200) # 创建一个matplotlib图形和轴对象
    if predicted_class == 1: # 如果预测为类别1 (有病)
        # 解释为什么预测为类别1
        shap.force_plot(explainer_shap.expected_value[1], 
                        shap_values[1][0], # SHAP值 for class 1, for the single sample
                        shap_input_df.iloc[0],    # 特征值 for the single sample
                        matplotlib=True, show=False, ax=ax) # 使用传入的ax，并且不立即显示
    else: # 如果预测为类别0 (无病)
        # 解释为什么预测为类别0
        shap.force_plot(explainer_shap.expected_value[0], 
                        shap_values[0][0], # SHAP值 for class 0, for the single sample
                        shap_input_df.iloc[0],    # 特征值 for the single sample
                        matplotlib=True, show=False, ax=ax) # 使用传入的ax，并且不立即显示
    
    # 将matplotlib图形显示在Streamlit中
    st.pyplot(fig, bbox_inches='tight')
    
# LIME Explanation
    st.subheader("LIME Explanation") # LIME解释部分的子标题
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,          # 使用加载的 X_test.csv 的 NumPy 数组作为背景数据
        feature_names=X_test.columns.tolist(),# 使用 X_test.csv 的列名作为特征名
                                              # 注意：这里应该与之前定义的 feature_names 列表保持一致
        class_names=['Not sick', 'Sick'],     # 类别名称，用于LIME输出的可读性
        mode='classification'                 # 表明是分类任务
    )
    
    # 为当前用户输入的样本生成LIME解释
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),            # 将用户输入的特征数组展平为1D，LIME期望这种格式
        predict_fn=model.predict_proba,       # 传递模型的预测概率函数
        num_features=len(feature_names)       # 显示所有特征的贡献
    )

    # 将LIME解释结果转换为HTML并在Streamlit中显示
    # show_table=False 避免在HTML中显示特征值的表格，使输出更简洁
    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True) # 使用Streamlit组件显示HTML内容