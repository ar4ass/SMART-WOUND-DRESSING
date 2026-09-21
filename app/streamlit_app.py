import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.calibration import metrics_to_table, train_and_evaluate_models
from src.color_analysis import compute_color_stats
from src.experimental_dataset import (
    ExperimentalDataError,
    generate_template,
    load_experimental_dataset,
)
from src.feature_extraction import extract_features
from src.history import add_entry, clear_history, read_history
from src.prediction import (
    ModelNotAvailableError,
    load_model,
    predict_from_features,
    save_model,
)
from src.preprocessing import ImageLoadError, load_image_from_bytes, preprocess_image
from src.roi import ROIError, get_center_roi, get_fixed_roi
from src.simulation_dataset import generate_simulation_dataset
from src.validation import (
    angle_variation_test,
    distance_variation_test,
    lighting_variation_test,
    repeatability_test,
    smartphone_variation_test,
)
from src.visualization import (
    plot_actual_vs_predicted,
    plot_channel_bar,
    plot_color_comparison,
)

DISCLAIMER = (
    "**Disclaimer.** Aplikasi ini adalah prototype perangkat lunak untuk keperluan "
    "penelitian dan edukasi. Sistem ini **bukan alat diagnosis klinis** dan tidak "
    "digunakan untuk menentukan kondisi luka atau penyakit. Material nanokomposit "
    "responsif pH masih dalam tahap pengembangan, sehingga belum tersedia data "
    "eksperimen maupun model kalibrasi tervalidasi."
)

st.set_page_config(page_title="SmartWound - Prototype", page_icon="🩹", layout="wide")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.sidebar.title("🩹 SmartWound")
page = st.sidebar.radio(
    "Menu",
    [
        "Home", "Image Analysis", "Result", "Calibration Demo",
        "Experimental Data", "History", "About",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Prototype software — bukan alat diagnosis klinis.")


def render_home():
    st.title("SmartWound")
    st.caption("Prototype software: monitoring perubahan warna dressing berbasis analisis citra smartphone")
    st.warning(DISCLAIMER)

    st.subheader("Tujuan Sistem")
    st.write(
        "SmartWound mengubah foto wound dressing menjadi data numerik (RGB/HSV) yang dapat "
        "dicatat dan dibandingkan dari waktu ke waktu, sebagai langkah awal menuju sistem "
        "monitoring yang lebih objektif. Material nanokomposit responsif pH yang menjadi "
        "dasar konsep ini masih dalam tahap pengembangan terpisah."
    )

    st.subheader("Status Saat Ini")
    st.info(
        "Software prototype telah dikembangkan (STEP 1-13), sementara material dressing "
        "responsif pH masih dalam tahap pengembangan/validasi eksperimen. Model kalibrasi "
        "yang tersedia di halaman *Calibration Demo* hanya dilatih dari **dataset simulasi**, "
        "bukan data eksperimen nyata."
    )


def render_image_analysis():
    st.title("Image Analysis")

    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Pilih gambar (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Belum ada gambar. Silakan unggah satu file gambar terlebih dahulu.")
        return

    try:
        loaded = load_image_from_bytes(uploaded_file.getvalue(), file_name=uploaded_file.name)
    except ImageLoadError as exc:
        st.error(str(exc))
        return

    st.image(loaded.image, caption=loaded.file_name, width=350)

    st.subheader("2. Preprocessing")
    processed = preprocess_image(loaded.array)
    st.caption(f"Diresize & denoise -> {processed.shape[1]}x{processed.shape[0]} px")

    st.subheader("3. ROI (Region of Interest)")
    roi_mode = st.radio("Mode ROI", ["Center crop (otomatis)", "Manual (atur sendiri)"], horizontal=True)

    try:
        if roi_mode.startswith("Center"):
            fraction = st.slider("Proporsi ROI dari sisi terpendek gambar", 0.1, 1.0, 0.5, 0.05)
            roi_result = get_center_roi(processed, fraction=fraction)
        else:
            h, w = processed.shape[:2]
            col_a, col_b = st.columns(2)
            with col_a:
                x = st.slider("X (kiri)", 0, max(w - 30, 0), 0)
                width = st.slider("Lebar ROI", 30, w - x, min(200, w - x))
            with col_b:
                y = st.slider("Y (atas)", 0, max(h - 30, 0), 0)
                height = st.slider("Tinggi ROI", 30, h - y, min(200, h - y))
            roi_result = get_fixed_roi(processed, x=x, y=y, width=width, height=height)
    except ROIError as exc:
        st.error(str(exc))
        return

    col_img, col_roi = st.columns(2)
    with col_img:
        st.caption("Gambar setelah preprocessing")
        st.image(processed, use_container_width=True)
    with col_roi:
        st.caption(f"ROI terpilih ({roi_result.width}x{roi_result.height}px)")
        st.image(roi_result.roi_array, use_container_width=True)

    st.subheader("4. Analyze (RGB / HSV / Fitur)")
    stats = compute_color_stats(roi_result.roi_array)
    features = extract_features(roi_result.roi_array)

    col_rgb, col_hsv = st.columns(2)
    with col_rgb:
        fig_rgb = plot_channel_bar(
            [stats.mean_r, stats.mean_g, stats.mean_b], ["R", "G", "B"], "Rata-rata RGB"
        )
        st.pyplot(fig_rgb)
    with col_hsv:
        fig_hsv = plot_channel_bar(
            [stats.mean_h, stats.mean_s, stats.mean_v], ["H", "S", "V"], "Rata-rata HSV"
        )
        st.pyplot(fig_hsv)

    with st.expander("Lihat semua nilai fitur (mean, std, normalized RGB)"):
        st.json(features)

    st.subheader("5. Estimasi (opsional, hanya jika model tersedia)")
    prediction_value = None
    prediction_model_name = None
    prediction_dataset_type = "no_model"

    model_candidates = [
        (name, dtype)
        for dtype in ("experimental", "simulation")
        for name in ("random_forest", "linear_regression", "polynomial_regression", "svr")
    ]
    loaded_model = None
    for name, dtype in model_candidates:
        try:
            loaded_model = load_model(name, dtype)
            prediction_model_name, prediction_dataset_type = name, dtype
            break
        except ModelNotAvailableError:
            continue

    if loaded_model is None:
        st.caption(
            "Belum ada model tersimpan. Latih model di halaman **Calibration Demo** "
            "(dataset simulasi) atau **Experimental Data** (dataset nyata, setelah tersedia)."
        )
    else:
        pred = predict_from_features(features, loaded_model, prediction_model_name, prediction_dataset_type)
        prediction_value = pred.value
        if pred.is_simulation_only:
            st.warning(
                f"Estimated value (SIMULATION MODEL ONLY): **{pred.value:.2f}**\n\n"
                "Ini BUKAN estimasi pH nyata -- model dilatih dari dataset simulasi, "
                "hanya untuk menguji alur software."
            )
        else:
            st.warning(
                f"Estimated value (model eksperimen: {prediction_model_name}): **{pred.value:.2f}**\n\n"
                "Berdasarkan model kalibrasi dari data eksperimen yang diunggah. "
                "Tetap bukan alat diagnosis klinis -- lihat disclaimer di halaman Home."
            )

    add_entry(
        image_name=loaded.file_name,
        features=features,
        predicted_value=prediction_value,
        model_name=prediction_model_name,
        dataset_type=prediction_dataset_type,
    )

    st.session_state.last_result = {
        "image": loaded.image,
        "roi_array": roi_result.roi_array,
        "stats": stats,
        "features": features,
        "prediction_value": prediction_value,
        "prediction_model_name": prediction_model_name,
        "dataset_type": prediction_dataset_type,
        "file_name": loaded.file_name,
    }

    st.success("Analisis selesai & tersimpan ke History. Lihat detail lengkap di halaman **Result**.")


def render_result():
    st.title("Result")

    result = st.session_state.last_result
    if result is None:
        st.info("Belum ada analisis. Buka halaman **Image Analysis** dan unggah gambar dulu.")
        return

    st.caption(f"Hasil analisis untuk: {result['file_name']}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(result["image"], caption="Gambar asli", use_container_width=True)
    with col2:
        st.image(result["roi_array"], caption="ROI yang dianalisis", use_container_width=True)

    stats = result["stats"]
    original_rgb = (int(stats.mean_r), int(stats.mean_g), int(stats.mean_b))
    fig = plot_color_comparison(original_rgb, original_rgb)
    st.pyplot(fig)

    st.subheader("RGB values")
    st.table(
        {
            "Channel": ["R", "G", "B"],
            "Mean": [round(stats.mean_r, 2), round(stats.mean_g, 2), round(stats.mean_b, 2)],
            "Std Dev": [round(stats.std_r, 2), round(stats.std_g, 2), round(stats.std_b, 2)],
        }
    )

    st.subheader("HSV values")
    st.table(
        {
            "Channel": ["H", "S", "V"],
            "Mean": [round(stats.mean_h, 2), round(stats.mean_s, 2), round(stats.mean_v, 2)],
            "Std Dev": [round(stats.std_h, 2), round(stats.std_s, 2), round(stats.std_v, 2)],
        }
    )

    st.subheader("Estimated Parameter")
    if result["prediction_value"] is None:
        st.info(
            "Experimental prediction unavailable.\n\n"
            "Current result is for software prototype/simulation only."
        )
    else:
        st.warning(
            f"Estimated value (model: {result['prediction_model_name']}, "
            f"dataset: {result['dataset_type']}): **{result['prediction_value']:.2f}** "
            "-- SIMULATION ONLY, bukan nilai pH nyata."
        )


def render_calibration_demo():
    st.title("Calibration Demo (Simulation Dataset)")
    st.error(
        "Halaman ini HANYA memakai dataset simulasi (data buatan/sintetis). "
        "Metrik MAE/RMSE/R2 di bawah TIDAK merepresentasikan performa sistem yang "
        "sesungguhnya dan TIDAK BOLEH dikutip sebagai hasil eksperimen."
    )

    n_samples = st.slider("Jumlah sampel simulasi", 30, 300, 150, 10)

    if st.button("Generate dataset simulasi & latih 4 model"):
        df = generate_simulation_dataset(n_samples=n_samples)
        st.dataframe(df.head(10))

        X = df[["R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean"]].values
        y = df["target_value"].values

        result = train_and_evaluate_models(X, y, dataset_type="simulation")
        st.session_state.calibration_result = result

        st.subheader("Perbandingan Model (data test, SIMULASI)")
        st.table(metrics_to_table(result))

        st.subheader(f"Actual vs Predicted -- model terbaik: {result.best_model_name}")
        best_model = result.models[result.best_model_name]
        y_pred_all = best_model.predict(X)
        fig = plot_actual_vs_predicted(y, y_pred_all, title=f"{result.best_model_name} (SIMULATION)")
        st.pyplot(fig)

        save_path = save_model(best_model, result.best_model_name, "simulation")
        st.success(
            f"Model terbaik ({result.best_model_name}) disimpan ke `{save_path}`. "
            "Sekarang halaman Image Analysis bisa menampilkan estimasi demo."
        )


def render_history():
    st.title("History")

    rows = read_history()
    if not rows:
        st.info("Belum ada riwayat analisis.")
        return

    st.dataframe(rows, use_container_width=True)

    if st.button("Hapus semua riwayat", type="secondary"):
        clear_history()
        st.success("Riwayat dihapus.")
        st.rerun()


def render_about():
    st.title("About")

    st.subheader("Latar Belakang")
    st.write(
        "SmartWound dikembangkan sebagai bagian dari proyek mahasiswa lintas bidang "
        "Nanoteknologi-Kesehatan-Informatika: dressing luka berbasis material nanokomposit "
        "responsif pH yang berubah warna, dipantau lewat analisis citra smartphone."
    )

    st.subheader("Computer Vision yang Dipakai")
    st.write(
        "Pipeline: preprocessing (resize & denoise) -> ROI -> ekstraksi RGB/HSV -> fitur "
        "warna -> (setelah data eksperimen tersedia) model kalibrasi -> hasil monitoring."
    )

    st.subheader("Batasan Sistem")
    st.markdown(
        "- Bukan alat diagnosis klinis, tidak mendeteksi infeksi.\n"
        "- Belum menggantikan pemeriksaan tenaga medis.\n"
        "- Model kalibrasi yang tersedia saat ini hanya dari data simulasi.\n"
        "- Sensitif terhadap kondisi pencahayaan (koreksi penuh belum diterapkan).\n"
        "- Belum ada metrik akurasi yang valid karena belum ada data eksperimen nyata."
    )

    st.subheader("Status Pengembangan")
    st.markdown(
        "1. Software Prototype -- **selesai** (upload, preprocessing, ROI, RGB/HSV, "
        "fitur, dataset simulasi, kalibrasi, prediksi, history)\n"
        "2. Infrastruktur data eksperimen (upload, validasi, kalibrasi, uji stabilitas) "
        "-- **selesai**, siap menerima data nyata kapan pun tersedia\n"
        "3. Smart Dressing (material nanokomposit) -- **belum**, pengembangan terpisah "
        "di luar cakupan software\n"
        "4. Experimental Dataset nyata -- **belum ada**, menunggu material siap\n"
        "5. Calibration & Validation dengan data nyata -- **menunggu poin 4**\n"
        "6. Mobile Application (Flutter) -- belum\n"
        "7. Prototype Integrated System (dressing + optical box + kamera + software) -- belum"
    )


def render_experimental_data():
    st.title("Experimental Data")
    st.info(
        "Halaman ini untuk data eksperimen NYATA dari smart dressing (bukan simulasi). "
        "Belum ada data eksperimen yang dikumpulkan sampai saat ini -- material "
        "nanokomposit masih dalam pengembangan terpisah."
    )

    st.subheader("1. Download Template CSV")
    template_path = "data/sample/experimental_template.csv"
    generate_template(template_path)
    with open(template_path, "rb") as f:
        st.download_button(
            "Download experimental_template.csv",
            data=f.read(),
            file_name="experimental_template.csv",
            mime="text/csv",
        )
    st.caption(
        "Kolom wajib: sample_id, date, material_id, pH_reference, image_path, "
        "R_mean, G_mean, B_mean, H_mean, S_mean, V_mean, lighting_condition, "
        "smartphone_model, distance_cm, angle, replicate."
    )

    st.subheader("2. Upload Dataset Eksperimen")
    uploaded_csv = st.file_uploader("Upload CSV data eksperimen", type=["csv"])

    if uploaded_csv is None:
        return

    tmp_path = f"data/raw/{uploaded_csv.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_csv.getvalue())

    try:
        df = load_experimental_dataset(tmp_path)
    except ExperimentalDataError as exc:
        st.error(f"Dataset tidak valid: {exc}")
        return

    st.success(f"Dataset valid: {len(df)} baris.")
    st.dataframe(df, use_container_width=True)

    st.subheader("3. Kalibrasi dari Data Eksperimen")
    if st.button("Latih 4 model dari data eksperimen ini"):
        X = df[["R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean"]].values
        y = df["pH_reference"].values

        if len(df) < 10:
            st.warning(
                f"Dataset masih sangat kecil ({len(df)} baris). Hasil kalibrasi di bawah "
                "ini secara statistik belum bisa dipercaya -- kumpulkan lebih banyak sampel."
            )

        result = train_and_evaluate_models(X, y, dataset_type="experimental")
        st.table(metrics_to_table(result))

        best_model = result.models[result.best_model_name]
        save_path = save_model(best_model, result.best_model_name, "experimental")
        st.success(f"Model terbaik ({result.best_model_name}) disimpan ke `{save_path}`.")

    st.subheader("4. Uji Stabilitas (Validation)")
    validation_tabs = st.tabs(["Repeatability", "Lighting", "Smartphone", "Distance", "Angle"])

    with validation_tabs[0]:
        st.dataframe(repeatability_test(df), use_container_width=True)
    with validation_tabs[1]:
        try:
            st.dataframe(lighting_variation_test(df), use_container_width=True)
        except ValueError as exc:
            st.warning(str(exc))
    with validation_tabs[2]:
        try:
            st.dataframe(smartphone_variation_test(df), use_container_width=True)
        except ValueError as exc:
            st.warning(str(exc))
    with validation_tabs[3]:
        try:
            st.dataframe(distance_variation_test(df), use_container_width=True)
        except ValueError as exc:
            st.warning(str(exc))
    with validation_tabs[4]:
        try:
            st.dataframe(angle_variation_test(df), use_container_width=True)
        except ValueError as exc:
            st.warning(str(exc))


PAGES = {
    "Home": render_home,
    "Image Analysis": render_image_analysis,
    "Result": render_result,
    "Calibration Demo": render_calibration_demo,
    "Experimental Data": render_experimental_data,
    "History": render_history,
    "About": render_about,
}

PAGES[page]()
