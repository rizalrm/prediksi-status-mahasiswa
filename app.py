import streamlit as st
import pandas as pd
import joblib

# Load model dan nama fitur
model = joblib.load('model_rf.pkl')
feature_names = joblib.load('feature_names.pkl')

# Mapping untuk kolom kategorikal
course_map = {
    "Biofuel Production Technologies": 33,
    "Animation and Multimedia Design": 171,
    "Social Service (Evening Attendance)": 8014,
    "Agronomy": 9003,
    "Communication Design": 9070,
    "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119,
    "Equinculture": 9130,
    "Management": 9147,
    "Social Service": 9238,
    "Tourism": 9254,
    "Nursing": 9500,
    "Oral Hygiene": 9556,
    "Advertising and Marketing Management": 9670,
    "Journalism and Communication": 9773,
    "Basic Education": 9853,
    "Management (Evening Attendance)": 9991
}

st.title("Prediksi Status Mahasiswa: Graduate / Dropout / Enrolled")
st.write("Silakan masukkan data mahasiswa berikut:")

# --- Contoh fitur yang akan diinput user ---
# Sisa fitur akan diisi default (0/rata2)
course = st.selectbox('Jurusan', list(course_map.keys()))
gender = st.radio('Jenis Kelamin', ['Male', 'Female'])
age = st.number_input('Umur saat mendaftar', min_value=15, max_value=60, value=20)
admission_grade = st.slider('Admission Grade', min_value=0, max_value=200, value=140)
scholarship = st.radio('Penerima Beasiswa', ['Ya', 'Tidak'])

# --- Data input user ke dictionary ---
input_dict = {
    'Course': course_map[course],
    'Gender': 1 if gender == 'Male' else 0,
    'Age_at_enrollment': age,
    'Admission_grade': admission_grade,
    'Scholarship_holder': 1 if scholarship == 'Ya' else 0,
}

# Bisa tambahkan fitur lain yang ingin diinput user

# --- Inisialisasi DataFrame dengan seluruh kolom seperti saat training ---
input_complete = pd.DataFrame(columns=feature_names)
input_complete.loc[0] = 0  # Default semua 0

# Isi dengan input user
for k, v in input_dict.items():
    if k in feature_names:
        input_complete.at[0, k] = v

# --- Prediksi ---
if st.button("Prediksi Status"):
    pred = model.predict(input_complete)[0]
    label_map = {0: 'Graduate', 1: 'Dropout', 2: 'Enrolled'}
    st.success(f"Status Mahasiswa: **{label_map[pred]}**")

