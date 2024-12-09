import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# تحميل البيانات
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv"
    data = pd.read_csv(url)
    return data

# معالجة البيانات
def preprocess_data(data):
    # معالجة القيم الفارغة
    data = data.dropna()
    # تحويل الأعمدة النصية إلى أرقام
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].astype('category').cat.codes
    return data

# تحميل البيانات
st.title("توقع درجات Final Score")
data = load_data()
st.write("## البيانات الأولية:")
st.write(data.head())

# معالجة البيانات
data = preprocess_data(data)

# تحديد المدخلات والمخرجات
X = data.drop("Final Score", axis=1)  # المدخلات
y = data["Final Score"]  # المخرجات

# تقسيم البيانات للتدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestClassifier()
model.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred = model.predict(X_test)

# عرض الدقة
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### دقة النموذج: {accuracy:.2f}")

# واجهة المستخدم لتوقع درجات جديدة
st.write("## إدخال بيانات الطالب:")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=0)

input_df = pd.DataFrame([input_data])

# توقع الدرجة النهائية
if st.button("توقع"):
    prediction = model.predict(input_df)
    st.write(f"### الدرجة النهائية المتوقعة: {prediction[0]}")
