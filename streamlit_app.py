import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# عنوان التطبيق
st.title('🎓 Student Final Exam Score Prediction App')

st.info('This app predicts the final exam score of students based on their performance scores.')

# تحميل البيانات
with st.expander('📊 Dataset'):
    # قراءة البيانات
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    
    # حذف العمود غير المستخدم
    df.drop(columns=['Total'], inplace=True)
    
    # معالجة البيانات المفقودة إن وجدت
    df.fillna(0, inplace=True)
    
    # عرض البيانات الأولية
    st.write('### Raw Data:')
    st.dataframe(df)

# تحليل البيانات
with st.expander('📊 Data Analysis'):
    st.write('### Correlation Matrix:')
    st.write(df.corr())
    
    st.write('### Pairplot (relationship between features):')
    st.line_chart(df[['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score', 'Final_Score']])

# تجهيز البيانات للتعلم الآلي
with st.expander('⚙️ Data Preparation'):
    # ترميز عمود الجنس
    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    # تحديد الميزات والهدف
    X = df_encoded.drop(columns=['Final_Score', 'Student_ID'])
    y = df['Final_Score']
    
    # التحقق من القيم غير الرقمية
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(y, errors='coerce').fillna(0)
    
    st.write('### Features (X):')
    st.dataframe(X)
    st.write('### Target (y):')
    st.dataframe(y)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تدريب النموذج
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# تقييم النموذج
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with st.expander('📊 Model Evaluation'):
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")

# واجهة المستخدم
with st.sidebar:
    st.header('🔍 Enter Student Data:')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    attendance_score = st.slider('Attendance Score', 0, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 15)

# تجهيز بيانات المستخدم
new_data = pd.DataFrame({
    'Attendance_Score': [attendance_score],
    'Mid_Exam_Score': [mid_exam_score],
    'Lab_Exam_Score': [lab_exam_score],
    'Activity_Score': [activity_score],
    'Gender_Male': [1 if gender == 'Male' else 0]
})

# التأكد من تطابق الأعمدة مع النموذج
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# التنبؤ بالدرجة النهائية
predicted_final_score = model.predict(new_data)[0]

# عرض التنبؤ
with st.expander('📈 Prediction Results'):
    st.write('### Predicted Final Exam Score:')
    st.success(f'The predicted final exam score is: **{predicted_final_score:.2f}**')

# عرض توزيع التوقعات
with st.expander('📊 Actual vs Predicted Final Scores'):
    scatter_data = pd.DataFrame({'Actual Final Score': y_test, 'Predicted Final Score': y_pred})
    st.line_chart(scatter_data)
