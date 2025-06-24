import streamlit as st
import pandas as pd
import joblib

# 1. Load semua file hasil training
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')      # List nama kolom X_train
mean_values = pd.read_pickle('mean_values.pkl')       # Series: mean dari X_train
fitur_numerik = joblib.load('fitur_numerik.pkl')      # List nama kolom numerik (fitur yang di-scaling)

# 2. Fitur penting untuk user input
fitur_penting = [
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Admission_grade',
    'Curricular_units_2nd_sem_evaluations',
    'Age_at_enrollment',
    'Previous_qualification_grade',
    'Curricular_units_1st_sem_evaluations',
    'Tuition_fees_up_to_date'
]

st.title("Prediksi Status Mahasiswa: Graduate / Dropout / Enrolled")
st.write("Isi fitur utama berikut untuk prediksi:")

# 3. Form input user
input_dict = {}
input_dict['Curricular_units_2nd_sem_approved'] = st.number_input(
    "Mata kuliah lulus semester 2", min_value=0, max_value=50, value=6)
input_dict['Curricular_units_2nd_sem_grade'] = st.number_input(
    "Nilai rata-rata semester 2", min_value=0.0, max_value=20.0, value=13.0)
input_dict['Curricular_units_1st_sem_approved'] = st.number_input(
    "Mata kuliah lulus semester 1", min_value=0, max_value=50, value=6)
input_dict['Curricular_units_1st_sem_grade'] = st.number_input(
    "Nilai rata-rata semester 1", min_value=0.0, max_value=20.0, value=13.0)
input_dict['Admission_grade'] = st.number_input(
    "Admission grade", min_value=0.0, max_value=200.0, value=130.0)
input_dict['Curricular_units_2nd_sem_evaluations'] = st.number_input(
    "Jumlah evaluasi semester 2", min_value=0, max_value=50, value=6)
input_dict['Age_at_enrollment'] = st.number_input(
    "Usia saat mendaftar", min_value=15, max_value=60, value=20)
input_dict['Previous_qualification_grade'] = st.number_input(
    "Nilai kualifikasi sebelumnya", min_value=0.0, max_value=200.0, value=120.0)
input_dict['Curricular_units_1st_sem_evaluations'] = st.number_input(
    "Jumlah evaluasi semester 1", min_value=0, max_value=50, value=6)
input_dict['Tuition_fees_up_to_date'] = st.radio(
    "Status pembayaran UKT (1 = Lunas, 0 = Belum Lunas)", [0, 1], index=1)

# 4. Inisialisasi DataFrame input satu baris, lengkap semua kolom
input_data = mean_values.copy()
input_df = pd.DataFrame([input_data], columns=feature_names)

# 5. Update nilai fitur penting dengan input user
for col in fitur_penting:
    input_df.at[0, col] = input_dict[col]

# 6. Scaling pada seluruh fitur numerik (harus urut & lengkap seperti training)
input_df[fitur_numerik] = scaler.transform(input_df[fitur_numerik])

# 7. Prediksi!
if st.button("Prediksi Status"):
    missing = set(feature_names) - set(input_df.columns)
    if missing:
        st.error(f"Missing columns in input_df: {missing}")
    else:
        y_pred = model.predict(input_df)[0]
        label_map = {0: "Graduate", 1: "Dropout", 2: "Enrolled"}
        st.success(f"Prediksi Status Mahasiswa: **{label_map[y_pred]}**")
        #st.write("Data Input (fitur penting):", input_df[fitur_penting])

# --- Opsional: cek urutan dan isi kolom ---
# st.write("feature_names:", feature_names)
# st.write("input_df.columns:", input_df.columns.tolist())
# st.write("input_df shape:", input_df.shape)
