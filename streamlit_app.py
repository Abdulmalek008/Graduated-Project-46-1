import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# إعداد عنوان التطبيق
st.title('🎓 Student Final Exam Score Prediction')

# وصف التطبيق
st.info('This app predicts the final exam score based on attendance, mid exam, lab exam, and activity scores.')

# تحميل البيانات
# يمكن أن يكون لديك مصدر بيانات محلي أو تحميل من ملف CSV
with st.expander('📊 Upload Student Data'):
    # تحميل البيانات من ملف CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('### Raw Data:')
        st.dataframe(df)

        # عرض الأعمدة المتاحة
        st.write('### Available Columns in the Data:')
        st.write(df.columns)

# تجهيز البيانات
with st.expander('⚙️ Data Preparation'):
    # عرض الأعمدة الفعلية التي تم تحميلها
    st.write('### Available Columns in the DataFrame:')
    st.write(df.columns)

    # هنا يمكنك تعديل الأعمدة التي تستخدمها بناءً على الأعمدة الفعلية في البيانات
    required_columns = ['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score', 'Final_Score']
    
    # التحقق إذا كانت الأعمدة المطلوبة موجودة في البيانات
    missing_columns = [col for col in required_columns if col not in df.columns]
    if len(missing_columns) > 0:
        st.warning(f"The following required columns are missing: {', '.join(missing_columns)}")
    else:
        # إزالة الأعمدة غير المهمة أو التي لن نستخدمها
        df = df.drop(columns=['Total_Score'], errors='ignore')

        # تقسيم البيانات إلى المدخلات (features) والهدف (target)
        X = df[['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score']]
        y = df['Final_Score']  # نهدف للتنبؤ بدرجة الفاينل سكور

        # عرض البيانات
        st.write('### Features (X):')
        st.dataframe(X)
        st.write('### Target (y):')
        st.dataframe(y)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تحسين البيانات باستخدام StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# استخدام نموذج RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# واجهة المستخدم للحصول على المدخلات
with st.sidebar:
    st.header('🔍 Enter Student Data:')
    attendance_score = st.slider('Attendance Score', 0, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 15)

# تجهيز بيانات المستخدم للتنبؤ
user_data = np.array([[attendance_score, mid_exam_score, lab_exam_score, activity_score]])
user_data_scaled = scaler.transform(user_data)

# التنبؤ بدرجة الفاينل سكور
predicted_final_score = model.predict(user_data_scaled)[0]

# التأكد من أن درجة الفاينل لا تتجاوز 40
predicted_final_score = min(predicted_final_score, 40)

# حساب المجموع الكلي من الدرجات المدخلة
total_score = attendance_score + mid_exam_score + lab_exam_score + activity_score + predicted_final_score

# تصنيف الطالب بناءً على المجموع الكلي
if total_score >= 80:
    grade = 'A'
elif total_score >= 60:
    grade = 'B'
else:
    grade = 'C'

# عرض النتائج
st.write(f"### Predicted Final Exam Score: {predicted_final_score:.2f}")
st.write(f"### Total Score: {total_score:.2f}")
st.write(f"### Predicted Grade: {grade}")

# عرض الجدول الذي يحتوي على البيانات المدخلة والتنبؤات
input_data = {
    'Attendance Score': [attendance_score],
    'Mid Exam Score': [mid_exam_score],
    'Lab Exam Score': [lab_exam_score],
    'Activity Score': [activity_score],
    'Predicted Final Exam Score': [predicted_final_score],
    'Total Score': [total_score],
    'Predicted Grade': [grade]
}

input_df = pd.DataFrame(input_data)

with st.expander('📊 Prediction Table'):
    st.write('### Entered Data and Predicted Grade:')
    st.dataframe(input_df)

# رسم بياني لتوزيع درجات الفاينل المتوقعة
with st.expander('📈 Prediction Distribution'):
    st.write('### Distribution of Predicted Final Exam Scores:')
    fig, ax = plt.subplots()
    ax.scatter(df['Attendance_Score'] + df['Mid_Exam_Score'] + df['Lab_Exam_Score'] + df['Activity_Score'], df['Final_Score'], color='blue', label='Actual Final Score')
    ax.scatter(total_score, predicted_final_score, color='red', label='Predicted Final Score', zorder=5)
    ax.set_xlabel('Total Performance (Attendance, Mid Exam, Lab Exam, Activity)')
    ax.set_ylabel('Final Exam Score')
    ax.legend()
    st.pyplot(fig)
