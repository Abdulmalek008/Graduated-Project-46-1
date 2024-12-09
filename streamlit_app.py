import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# عنوان التطبيق
st.title('🎓 Student Grade Prediction App')

# تحميل البيانات
with st.expander('📊 Dataset'):
    # قراءة البيانات
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    
    # حذف العمود غير المستخدم
    df.drop(columns=['Total'], inplace=True)

    # تنظيف البيانات: التأكد من أن جميع الأعمدة الرقمية قابلة للاستخدام
    st.write('### Data Information:')
    st.write(df.info())

    # اختيار الأعمدة الرقمية فقط وتنظيف البيانات
    df_cleaned = df.select_dtypes(include=[np.number])  # اختيار الأعمدة الرقمية فقط
    df_cleaned = df_cleaned.fillna(0)  # ملء القيم المفقودة بـ 0

    # حساب مصفوفة الارتباط
    st.write('### Correlation Matrix:')
    st.write(df_cleaned.corr())
    
    st.write('### Cleaned Data:')
    st.dataframe(df_cleaned)

# إضافة عمود الدرجات النهائية
def calculate_level(row):
    total_score = row['Attendance_Score'] + row['Mid_Exam_Score'] + row['Lab_Exam_Score'] + row['Activity_Score']
    if total_score >= 80:
        return 'A'
    elif total_score >= 60:
        return 'B'
    else:
        return 'C'

df['Level'] = df.apply(calculate_level, axis=1)

# تجهيز البيانات للتعلم الآلي
with st.expander('⚙️ Data Preparation'):
    # تحديد الميزات والهدف
    X = df[['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score']]
    y = df['Final_Score']
    
    st.write('### Features (X):')
    st.dataframe(X)
    st.write('### Target (y):')
    st.dataframe(y)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تدريب النموذج للتنبؤ بالفاينال سكور
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# واجهة المستخدم
with st.sidebar:
    st.header('🔍 Enter Student Data:')
    attendance_score = st.slider('Attendance Score', 0, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 15)

# تجهيز بيانات المستخدم
new_data = pd.DataFrame({
    'Attendance_Score': [attendance_score],
    'Mid_Exam_Score': [mid_exam_score],
    'Lab_Exam_Score': [lab_exam_score],
    'Activity_Score': [activity_score]
})

# التنبؤ بالفاينال سكور
predicted_final_score = model.predict(new_data)[0]
st.write(f"### Predicted Final Exam Score: {predicted_final_score:.2f}")

# حساب المجموع الكلي
total_score = attendance_score + mid_exam_score + lab_exam_score + activity_score + predicted_final_score

# تصنيف المستوى النهائي
if total_score >= 80:
    level = 'A'
elif total_score >= 60:
    level = 'B'
else:
    level = 'C'

# عرض النتائج
st.write(f"### Total Score: {total_score:.2f}")
st.write(f"### Predicted Grade: {level}")

# إنشاء جدول يعرض البيانات المدخلة والتنبؤ
input_data = {
    'Attendance Score': [attendance_score],
    'Mid Exam Score': [mid_exam_score],
    'Lab Exam Score': [lab_exam_score],
    'Activity Score': [activity_score],
    'Predicted Final Score': [predicted_final_score],
    'Total Score': [total_score],
    'Predicted Grade': [level]
}

input_df = pd.DataFrame(input_data)

# عرض الجدول للمستخدم
with st.expander('📊 Prediction Table'):
    st.write('### Entered Data and Predicted Grade:')
    st.dataframe(input_df)
