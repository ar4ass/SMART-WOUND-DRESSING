# System Architecture

```text
Smart Dressing -> Color Response -> Smartphone Camera -> Image Processing ->
ROI -> RGB/HSV -> Feature Extraction -> Calibration Model -> Monitoring Result
```

| Tahap | Modul | Status |
|---|---|---|
| Image load & preprocessing | src/preprocessing.py | selesai |
| ROI | src/roi.py | selesai (fixed/center crop) |
| RGB / HSV | src/color_analysis.py | selesai |
| Feature extraction | src/feature_extraction.py | selesai |
| Dataset simulasi | src/simulation_dataset.py | selesai |
| Visualisasi | src/visualization.py | selesai |
| Calibration model | src/calibration.py | selesai (jalan untuk simulasi & eksperimen) |
| Prediction | src/prediction.py | selesai |
| History | src/history.py | selesai (CSV lokal) |
| Skema & loader data eksperimen | src/experimental_dataset.py | selesai (infrastruktur, belum ada data) |
| Uji stabilitas (repeatability/lighting/dll) | src/validation.py | selesai (infrastruktur, belum ada data) |
| UI | app/streamlit_app.py | selesai, 7 halaman |

## Yang tidak bisa diselesaikan lewat software
- Material nanokomposit responsif pH nyata (di luar cakupan kode).
- Data eksperimen nyata (pH_reference terukur dari sampel asli).
- Optical box / hardware pengambilan gambar terstandar.
- Validasi lapangan (repeatability, lighting, dll) dengan data sungguhan --
  kodenya sudah siap, tinggal butuh datanya.
