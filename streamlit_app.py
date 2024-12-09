import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# تحميل البيانات
data = {
    "Student_ID": ["S001", "S002", "S003", "S004", "S005", "S006", "S007", "S008", "S009", "S010", "S011", "S012", "S013", "S014", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S022", "S023", "S024", "S025", "S026", "S027", "S028", "S029", "S030", "S031", "S032", "S033", "S034", "S035", "S036", "S037", "S038", "S039", "S040", "S041", "S042", "S043", "S044", "S045", "S046", "S047", "S048", "S049", "S050", "S051", "S052", "S053", "S054", "S055", "S056", "S057", "S058", "S059", "S060", "S061", "S062", "S063", "S064", "S065", "S066", "S067", "S068", "S069", "S070", "S071", "S072", "S073", "S074", "S075", "S076", "S077", "S078", "S079", "S080", "S081", "S082", "S083", "S084", "S085", "S086", "S087", "S088", "S089", "S090", "S091", "S092", "S093", "S094", "S095", "S096", "S097", "S098", "S099", "S100", "S101", "S102", "S103", "S104", "S105", "S106", "S107", "S108", "S109", "S110", "S111", "S112", "S113", "S114", "S115", "S116", "S117", "S118", "S119", "S120", "S121", "S122", "S123", "S124", "S125", "S126", "S127", "S128", "S129", "S130", "S131", "S132", "S133", "S134", "S135", "S136", "S137", "S138", "S139", "S140", "S141", "S142", "S143", "S144", "S145", "S146", "S147", "S148", "S149", "S150"],
    "Attendance_Score": [3, 3, 1, 1, 4, 2, 4, 1, 3, 5, 5, 4, 2, 2, 5, 4, 5, 4, 5, 4, 1, 4, 3, 1, 5, 1, 4, 4, 5, 2, 3, 1, 3, 3, 5, 4, 4, 5, 2, 4, 4, 5, 2, 4, 5, 2, 4, 4, 5, 5, 1, 4, 1, 5, 5, 4, 5, 2, 1, 4, 2, 4, 4, 3, 2, 5, 4, 5, 3, 5, 5, 4, 5, 4, 3, 1, 3, 3, 4, 5, 3, 1, 4, 3, 1, 2, 3, 5, 2, 5, 4, 5, 4, 2, 5, 2, 5, 5, 1, 2, 1, 2, 3, 3, 4, 5, 3, 4, 2, 4, 5, 4],
    "Mid_Exam_Score": [5, 1, 13, 2, 12, 2, 9, 6, 15, 13, 7, 6, 12, 13, 15, 9, 2, 3, 10, 2, 5, 5, 15, 5, 1, 10, 6, 3, 9, 11, 11, 2, 13, 8, 11, 9, 15, 9, 2, 7, 11, 7, 8, 11, 14, 12, 9, 3, 14, 1, 7, 2, 14, 1, 12, 15, 14, 2, 4, 9, 13, 12, 15, 5, 4, 2, 3, 14, 11, 13, 4, 1, 9, 13, 8, 4, 10, 9, 2, 13, 9, 6, 5, 12, 12, 6, 3, 11, 9, 2, 7, 11, 15, 11, 5, 14, 6, 5, 14, 7, 15, 2, 8, 9, 12, 6, 12, 11, 9, 12, 13, 4],
    "Lab_Exam_Score": [7, 15, 12, 11, 2, 4, 1, 8, 10, 6, 5, 12, 14, 11, 3, 10, 7, 12, 3, 6, 12, 6, 13, 14, 4, 9, 13, 1, 15, 9, 4, 5, 2, 7, 9, 4, 15, 4, 9, 10, 14, 15, 7, 5, 12, 14, 12, 10, 14, 4, 2, 5, 4, 6, 14, 15, 14, 14, 11, 12, 3, 6, 1, 2, 5, 15, 9, 4, 9, 5, 8, 7, 2, 13, 4, 4, 10, 13, 9, 1, 9, 7, 9, 9, 15, 2, 7, 13, 6, 10, 9, 4, 12, 10, 13, 9, 12, 2, 12, 12, 15, 15, 15, 6, 8, 3, 4, 10, 4],
    "Activity_Score": [7, 19, 5, 11, 19, 7, 20, 12, 21, 3, 12, 14, 16, 8, 5, 2, 8, 8, 4, 2, 25, 3, 10, 6, 14, 11, 22, 23, 9, 8, 21, 15, 12, 19, 14, 17, 10, 17, 8, 3, 8, 2, 3, 22, 5, 3, 22, 16, 9, 16, 23, 14, 20, 17, 8, 10, 6, 23, 1, 13, 6, 8, 19, 15, 12, 14, 11, 12, 7, 15, 6, 8, 7, 14, 2, 10, 10, 15, 9, 9, 13, 14, 3, 7, 13, 15, 8, 10, 13, 5, 14, 7, 9, 14, 11, 12, 6, 16, 3, 11, 7, 10, 8],
    "Gender": ["Male", "Male", "Female", "Male", "Female", "Female", "Male", "Female", "Female", "Female", "Female", "Male", "Female", "Female", "Male", "Male", "Female", "Male", "Female", "Female", "Female", "Male", "Male", "Female", "Female", "Male", "Female", "Male", "Male", "Male", "Female", "Male", "Female", "Female", "Male", "Male", "Female", "Male", "Female", "Female", "Female", "Female", "Female", "Male", "Female", "Male", "Female", "Female", "Male", "Female", "Female", "Female", "Male", "Female", "Female", "Female", "Female", "Female", "Male", "Female", "Female", "Female", "Male", "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Male", "Male", "Female", "Male", "Male", "Female", "Male", "Male", "Female", "Male", "Male", "Male", "Female", "Male", "Female", "Female", "Male", "Female", "Female", "Female", "Male", "Female", "Female", "Male", "Male", "Male", "Female", "Male", "Female", "Female", "Female", "Male", "Male", "Female", "Female", "Male", "Female", "Female", "Female", "Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Female", "Male", "Male", "Male", "Female", "Female", "Female", "Male", "Female", "Male", "Female", "Female", "Male"],
    "Total": [62, 73, 58, 60, 64, 31, 40, 66, 60, 37, 30, 68, 84, 43, 67, 37, 61, 46, 34, 49, 47, 27, 42, 66, 57, 35, 81, 37, 51, 47, 78, 28, 70, 16, 44, 36, 44, 37, 72, 51, 46, 55, 75, 65, 56, 48, 57, 42, 47, 40, 64, 25, 59, 60, 67, 50, 39, 71, 72, 84, 59, 79, 61, 59, 57, 43, 60, 57, 51, 72, 62, 70, 65, 66, 63, 63, 61, 61, 56, 58, 61, 60, 58, 57, 71, 53, 62, 64, 57, 57, 64, 55, 43, 75, 66, 60, 77, 62, 80, 68, 68, 57, 79, 60, 48, 46, 56, 69, 56],
    "Level": ["B", "B", "C", "B", "B", "C", "C", "B", "B", "C", "C", "B", "A", "C", "B", "C", "B", "C", "B", "C", "C", "C", "C", "B", "C", "C", "A", "C", "B", "C", "B", "C", "B", "C", "A", "B", "B", "C", "B", "C", "B", "C", "C", "C", "B", "B", "B", "B", "C", "B", "B", "C", "C", "C", "B", "C", "C", "B", "B", "B", "C", "C", "B", "B", "C", "C", "B", "B", "B", "A", "B", "B", "C", "B", "C", "B", "C", "C", "B", "A", "B", "B", "C", "C", "B", "B", "C", "B", "B", "C", "C", "B", "B", "B", "C", "B", "C", "C", "B", "A", "B", "C", "A", "B", "B", "A", "B", "C", "B", "C", "C", "B", "C", "C", "B", "A", "B", "B", "C"]
}

df = pd.DataFrame(data)

# حذف عمود Final_Score
X = df.drop(columns=["Final_Score", "Student_ID", "Level"])
y = df["Final_Score"]

# تحويل الأعمدة غير الرقمية إلى أرقام باستخدام التشفير (مثل Gender)
X = pd.get_dummies(X)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب نموذج RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التنبؤ بالدرجات
y_pred = model.predict(X_test)

# حساب الخطأ المطلق
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Mean Absolute Error: {mae}")

# التنبؤ بدرجة Final_Score
new_data = {
    "Attendance_Score": 3,
    "Mid_Exam_Score": 10,
    "Lab_Exam_Score": 12,
    "Activity_Score": 8,
    "Gender": "Female",
    "Total": 75
}

new_df = pd.DataFrame([new_data])
new_df = pd.get_dummies(new_df)

# التأكد من تطابق الأعمدة بين new_df و X
new_df = new_df.reindex(columns=X.columns, fill_value=0)

# التنبؤ بالدرجة النهائية
predicted_final_score = model.predict(new_df)
st.write(f"Predicted Final Score: {predicted_final_score[0]}")
