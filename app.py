
import gradio as gr
import pandas as pd
import joblib

# Load trained model
model_pipeline = joblib.load(
    "model/dropout_model.pkl"
)


def get_risk_level(probability):

    if probability < 0.40:
        return "Low Risk"

    elif probability < 0.70:
        return "Medium Risk"

    else:
        return "High Risk"


def predict_student(
    age,
    gender,
    family_income,
    internet_access,
    study_hours,
    attendance,
    assignment_delay,
    travel_time,
    part_time_job,
    scholarship,
    stress,
    gpa,
    semester_gpa,
    cgpa,
    semester,
    department,
    parental_education
):

    student = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Family_Income': family_income,
        'Internet_Access': internet_access,
        'Study_Hours_per_Day': study_hours,
        'Attendance_Rate': attendance,
        'Assignment_Delay_Days': assignment_delay,
        'Travel_Time_Minutes': travel_time,
        'Part_Time_Job': part_time_job,
        'Scholarship': scholarship,
        'Stress_Index': stress,
        'GPA': gpa,
        'Semester_GPA': semester_gpa,
        'CGPA': cgpa,
        'Semester': semester,
        'Department': department,
        'Parental_Education': parental_education
    }])

    prediction = model_pipeline.predict(student)[0]

    probability = model_pipeline.predict_proba(student)[0][1]

    risk = get_risk_level(probability)

    if prediction == 1:
        predicted_class = "Dropout"
    else:
        predicted_class = "Not Dropout"

    return (
        f"{probability * 100:.2f}%",
        risk,
        predicted_class
    )


interface = gr.Interface(

    fn=predict_student,

    inputs=[

        gr.Number(label="Age"),

        gr.Dropdown(
            ["Male", "Female"],
            label="Gender"
        ),

        gr.Number(label="Family Income"),

        gr.Dropdown(
            ["Yes", "No"],
            label="Internet Access"
        ),

        gr.Number(
            label="Study Hours per Day"
        ),

        gr.Number(
            label="Attendance Rate"
        ),

        gr.Number(
            label="Assignment Delay Days"
        ),

        gr.Number(
            label="Travel Time Minutes"
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Part-Time Job"
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Scholarship"
        ),

        gr.Number(
            label="Stress Index"
        ),

        gr.Number(
            label="GPA"
        ),

        gr.Number(
            label="Semester GPA"
        ),

        gr.Number(
            label="CGPA"
        ),

        gr.Dropdown(
            [
                "1st",
                "2nd",
                "3rd",
                "4th",
                "5th",
                "6th",
                "7th",
                "8th"
            ],
            label="Semester"
        ),

        gr.Textbox(
            label="Department"
        ),

        gr.Dropdown(
            [
                "Primary",
                "Secondary",
                "Bachelor",
                "Master",
                "PhD"
            ],
            label="Parental Education"
        )
    ],

    outputs=[

        gr.Textbox(
            label="Dropout Probability"
        ),

        gr.Textbox(
            label="Risk Level"
        ),

        gr.Textbox(
            label="Prediction"
        )
    ],

    title="Student Dropout Risk Prediction",

    description=(
        "Enter student information to estimate "
        "dropout probability and risk level."
    )
)


interface.launch(share=True)
