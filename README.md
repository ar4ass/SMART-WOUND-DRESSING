# SmartWound

Prototype perangkat lunak untuk **monitoring perubahan warna wound dressing** menggunakan analisis citra smartphone (RGB/HSV), dirancang untuk nantinya diintegrasikan dengan material dressing nanokomposit responsif pH.

> **Status saat ini:** Software prototype has been developed (STEP 1-15 dari roadmap internal), while the pH-responsive nanocomposite dressing is still under development/experimental validation. Model kalibrasi yang tersedia di aplikasi hanya dilatih dari **dataset simulasi**, bukan data eksperimen nyata.

---

## 1. Problem

Perubahan kondisi wound dressing saat ini umumnya hanya dinilai secara visual/subjektif. Diperlukan cara yang lebih objektif dan terdokumentasi untuk mencatat perubahan warna dressing dari waktu ke waktu.

## 2. Proposed Solution

Smart dressing responsif pH (mengubah warna) + analisis citra smartphone untuk mengubah perubahan warna tersebut menjadi data numerik yang dapat dicatat, divisualisasikan, dan (setelah kalibrasi dengan data nyata) diestimasi parameternya.

## 3. Disclaimer

Sistem ini **bukan alat diagnosis klinis**, tidak mendeteksi infeksi, dan tidak menggantikan pemeriksaan tenaga medis. Sistem hanya melakukan monitoring perubahan warna dan estimasi parameter berbasis kalibrasi — yang saat ini masih berbasis data simulasi, ditandai jelas di setiap tempat yang relevan di aplikasi.

---

## 4. System Architecture

Lihat [`docs/system_architecture.md`](docs/system_architecture.md) untuk detail pemetaan tiap tahap ke modul kode.

```text
Smart Dressing -> Color Response -> Smartphone Camera -> Image Processing ->
ROI -> RGB/HSV -> Feature Extraction -> Calibration Model -> Monitoring Result
```

---

## 5. Struktur Repository

```text
smartwound/
├── README.md
├── requirements.txt
├── .gitignore
├── app/
│   └── streamlit_app.py      # UI 6 halaman: Home, Image Analysis, Result,
│                              # Calibration Demo, History, About
├── src/
│   ├── preprocessing.py       # load_image_from_bytes, preprocess_image
│   ├── roi.py                  # get_center_roi, get_fixed_roi
│   ├── color_analysis.py       # compute_color_stats (RGB & HSV)
│   ├── feature_extraction.py   # extract_features (+ delta vs baseline)
│   ├── simulation_dataset.py   # generate_simulation_dataset (data_type=simulation)
│   ├── experimental_dataset.py # skema data eksperimen nyata, loader + validasi
│   ├── validation.py           # repeatability/lighting/smartphone/distance/angle test
│   ├── visualization.py        # plot_color_comparison, plot_channel_bar, plot_actual_vs_predicted
│   ├── calibration.py          # train_and_evaluate_models (4 model + metrik)
│   ├── prediction.py           # save_model, load_model, predict_from_features
│   └── history.py              # add_entry, read_history, clear_history
├── data/{raw,processed,sample}/
├── models/                   # model .joblib tersimpan (dibuat lewat halaman Calibration Demo)
├── notebooks/                # eksplorasi data & training (draft)
├── tests/                    # 44 unit test (pytest), semua modul di atas
├── docs/{system_architecture,methodology,validation}.md
└── assets/screenshots/
```

---

## 6. Installation

```bash
git clone <url-repository-anda>
cd smartwound

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 7. Run

```bash
python -m streamlit run app/streamlit_app.py
```

Buka `http://localhost:8501`. Alur cepat mencoba semua fitur:
1. Halaman **Calibration Demo** → klik "Generate dataset simulasi & latih 4 model" (sekali saja).
2. Halaman **Image Analysis** → upload foto apa pun → lihat ROI, RGB/HSV, dan estimasi bertanda SIMULATION ONLY.
3. Halaman **Result** → detail lengkap analisis terakhir.
4. Halaman **Experimental Data** → download template CSV, isi dengan data eksperimen nyata (kalau sudah ada), upload, latih model, lihat uji stabilitas.
5. Halaman **History** → semua analisis yang pernah dijalankan.

## 8. Testing

```bash
python -m pytest tests/ -v
```

57 test mencakup: image loading, preprocessing, ROI, RGB/HSV, feature extraction, dataset simulasi, dataset eksperimen, uji stabilitas, visualisasi, kalibrasi, prediksi, dan history.

---

## 9. Dataset

| Jenis | Status | Keterangan |
|---|---|---|
| Simulated dataset | **Tersedia** (`src/simulation_dataset.py`) | Data buatan berlabel `data_type=simulation`, hanya untuk menguji pipeline & UI. |
| Experimental dataset | Belum tersedia | Baru dibuat setelah smart dressing dan pengukuran pH referensi tersedia. Skema kolom sudah disiapkan (lihat `docs/methodology.md`). |

**Simulation Data ≠ Experimental Data.**

---

## 10. Limitations

- Belum merupakan alat diagnosis dan tidak menggantikan pemeriksaan klinis.
- Membutuhkan validasi material nanokomposit.
- Model kalibrasi saat ini hanya dilatih dari data simulasi — metrik MAE/RMSE/R2 di halaman Calibration Demo **tidak** mewakili performa nyata.
- Sensitif terhadap kondisi pencahayaan; koreksi penuh (white balance) belum diterapkan.
- ROI masih manual/center-crop, belum deteksi objek otomatis.
- History disimpan lokal (CSV), belum ada backend multi-pengguna.

---

## 11. Roadmap

| Versi | Isi | Status |
|---|---|---|
| V0.1 | Upload image → ROI → RGB → HSV | selesai |
| V0.2 | Simulated dataset → prototype result | selesai |
| V0.3 | Experimental image → kalibrasi → validasi | **infrastruktur selesai** (halaman "Experimental Data": upload, validasi, kalibrasi, uji stabilitas) — tinggal menunggu data eksperimen nyata untuk diisi |
| V1.0 | Integrated prototype (dressing + optical box + kamera + model + aplikasi) | belum — butuh material & hardware fisik, di luar cakupan software |

**Ini adalah batas akhir yang bisa diselesaikan lewat software saja.** Begitu smart dressing fisik dan data eksperimen (pH_reference terukur dari sampel asli) tersedia, tinggal: (1) isi template CSV di halaman Experimental Data, (2) upload, (3) klik latih model, (4) cek tab uji stabilitas. Tidak perlu tulis kode baru.

## 12. License

Belum ditentukan (rencana: MIT).
