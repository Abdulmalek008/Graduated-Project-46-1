import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# عنوان التطبيق
st.title('🎓 Student Final Exam Score Prediction App')

st.info('This app predicts the final exam score of students based on their performance scores in attendance, mid exam, lab exam, and activity.')

# تحميل البيانات
with st.expander('📊 Dataset'):
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    
    # عرض البيانات
    st.write('### Raw Data:')
    st.dataframe(df)

# تحديد الفئات بناءً على مجموع الدرجات
def grade_category(total_score):
    if total_score >= 80:
        return 'A'
    elif total_score >= 60:
        return 'B'
    else:
        return 'C'

# تجهيز البيانات
df['Total'] = df['Attendance_Score'] + df['Mid_Exam_Score'] + df['Lab_Exam_Score'] + df['Activity_Score']
df['Grade'] = df['Total'].apply(grade_category)

X = df[['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score']]
y = df['Grade']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# استخدام RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# واجهة المستخدم
with st.sidebar:
    st.header('🔍 Enter Student Data:')
    attendance_score = st.slider('Attendance Score', 0, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 15)

# تجهيز بيانات المستخدم للتنبؤ
new_data = pd.DataFrame({
    'Attendance_Score': [attendance_score],
    'Mid_Exam_Score': [mid_exam_score],
    'Lab_Exam_Score': [lab_exam_score],
    'Activity_Score': [activity_score]
})

# التنبؤ بالدرجة
predicted_grade = model.predict(new_data)[0]

# حساب المجموع الكلي من الدرجات المدخلة
total_score = attendance_score + mid_exam_score + lab_exam_score + activity_score

# عرض النتائج
st.write(f"### Predicted Grade: {predicted_grade}")
st.write(f"### Total Score: {total_score:.2f}")

# إنشاء جدول يعرض البيانات المدخلة والتنبؤ
input_data = {
    'Attendance Score': [attendance_score],
    'Mid Exam Score': [mid_exam_score],
    'Lab Exam Score': [lab_exam_score],
    'Activity Score': [activity_score],
    'Predicted Grade': [predicted_grade],
    'Total Score': [total_score]
}

input_df = pd.DataFrame(input_data)

# عرض الجدول للمستخدم
with st.expander('📊 Prediction Table'):
    st.write('### Entered Data and Predicted Grade:')
    st.dataframe(input_df)

# رسم بياني لتوزيع الدرجات
with st.expander('📈 Prediction Distribution'):
    st.write('### Distribution of Predicted Grades:')
    fig, ax = plt.subplots()
    ax.scatter(df['Attendance_Score'] + df['Mid_Exam_Score'] + df['Lab_Exam_Score'] + df['Activity_Score'], df['Grade'], color='blue', label='Actual Grades')
    ax.scatter(total_score, predicted_grade, color='red', label='Predicted Grade', zorder=5)
    ax.set_xlabel('Total Performance (Attendance, Mid Exam, Lab Exam, Activity)')
    ax.set_ylabel('Grade')
    ax.legend()
    st.pyplot(fig)
