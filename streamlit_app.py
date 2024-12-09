import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the dataset from the provided URL
url = "https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv"
data = pd.read_csv(url)

# Drop the 'Final_Score' column as we are predicting it
X = data.drop(columns=["Final_Score", "Student_ID"])  # Features excluding Final_Score
y = data["Final_Score"]  # Target variable is Final_Score

# Encoding categorical data
X = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting the Final Score on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Display the results
st.write(f"Mean Absolute Error: {mae}")
st.write("Model has been trained. You can now predict Final Score.")

# User Input for Prediction
attendance = st.number_input("Attendance Score", min_value=0, max_value=5)
mid_exam = st.number_input("Mid Exam Score", min_value=0, max_value=40)
lab_exam = st.number_input("Lab Exam Score", min_value=0, max_value=40)
activity_score = st.number_input("Activity Score", min_value=0, max_value=40)
gender = st.selectbox("Gender", options=["Male", "Female"])
total = st.number_input("Total", min_value=0, max_value=100)
level = st.selectbox("Level", options=["A", "B", "C"])

# Prepare the input data for prediction
input_data = pd.DataFrame({
    "Attendance_Score": [attendance],
    "Mid_Exam_Score": [mid_exam],
    "Lab_Exam_Score": [lab_exam],
    "Activity_Score": [activity_score],
    "Gender_Female": [1 if gender == "Female" else 0],
    "Total": [total],
    "Level_B": [1 if level == "B" else 0],
    "Level_C": [1 if level == "C" else 0]
})

# Predict the Final Score based on the user input
final_score_pred = model.predict(input_data)
st.write(f"Predicted Final Score: {final_score_pred[0]:.2f}")
