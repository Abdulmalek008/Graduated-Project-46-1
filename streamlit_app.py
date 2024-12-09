import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# تحميل البيانات من الرابط
url = 'https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv'
df = pd.read_csv(url)

# عرض البيانات
st.write("Dataset:", df)

# التأكد من الأعمدة المطلوبة
if all(col in df.columns for col in ['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score', 'Final_Score']):
    
    # تقسيم البيانات إلى X (المدخلات) و y (الهدف)
    X = df[['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score']]
    y = df['Final_Score']
    
    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # تدريب نموذج RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # التنبؤ بدرجات الفاينال
    predicted_final_score = model.predict(X_test)
    
    # حساب الخطأ
    mae = mean_absolute_error(y_test, predicted_final_score)
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    
    # تصنيف الطلاب بناءً على الدرجة النهائية المتوقعة
    def grade_classification(final_score):
        if final_score >= 80:
            return 'A'
        elif final_score >= 60:
            return 'B'
        else:
            return 'C'
    
    # عرض الرسم البياني
    st.write("Predicted Final Exam Scores vs Actual Final Scores")
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_test, y=predicted_final_score)
    plt.xlabel('Actual Final Score')
    plt.ylabel('Predicted Final Score')
    plt.title('Comparison of Predicted and Actual Final Exam Scores')
    st.pyplot()
    
    # تنبؤ مع عرض التصنيف
    predicted_final_score_avg = np.mean(predicted_final_score)
    predicted_grade = grade_classification(predicted_final_score_avg)
    
    st.write(f"Predicted Final Exam Score: {predicted_final_score_avg:.2f}")
    st.write(f"Predicted Grade: {predicted_grade}")
else:
    st.write("The necessary columns are not present in the dataset.")
