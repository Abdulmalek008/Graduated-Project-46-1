import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# عنوان التطبيق
st.title('🎓 Student Grade Prediction App')

# تحميل البيانات
with st.expander('📊 Dataset'):
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    df.drop(columns=['Total'], inplace=True)

    # تنظيف البيانات
    df_cleaned = df.select_dtypes(include=[np.number]).fillna(0)
    st.write('### Correlation Matrix:')
    st.write(df_cleaned.corr())
    st.write('### Dataset:')
    st.dataframe(df)

# تطبيع البيانات
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df[['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score']])
df_normalized = pd.DataFrame(normalized_features, columns=['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score'])
df['Normalized_Final_Score'] = df['Final_Score'] / 40  # تطبيع الفاينل ضمن النطاق [0, 1]

# تقسيم البيانات
X = df_normalized
y = df['Normalized_Final_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تدريب النموذج
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# واجهة المستخدم
with st.sidebar:
    st.header('🔍 Enter Student Data:')
    attendance_score = st.slider('Attendance Score', 0, 5, 5)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 15)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 15)
    activity_score = st.slider('Activity Score', 0, 25, 25)

# تجهيز بيانات المستخدم
input_data = pd.DataFrame({
    'Attendance_Score': [attendance_score],
    'Mid_Exam_Score': [mid_exam_score],
    'Lab_Exam_Score': [lab_exam_score],
    'Activity_Score': [activity_score]
})

# تطبيع المدخلات
normalized_input = scaler.transform(input_data)
predicted_normalized_final_score = model.predict(normalized_input)[0]

# إعادة حساب الدرجة النهائية المتوقعة (من 40)
predicted_final_score = predicted_normalized_final_score * 40

# حساب المجموع الكلي
total_score = attendance_score + mid_exam_score + lab_exam_score + activity_score + predicted_final_score

# تصنيف الدرجة
if total_score >= 80:
    grade = 'A'
elif total_score >= 60:
    grade = 'B'
else:
    grade = 'C'

# عرض النتائج
st.write(f"### Predicted Final Exam Score: {predicted_final_score:.2f} / 40")
st.write(f"### Total Score: {total_score:.2f} / 100")
st.write(f"### Predicted Grade: {grade}")
