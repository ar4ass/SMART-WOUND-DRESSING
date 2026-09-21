# Methodology

## Dataset simulasi (src/simulation_dataset.py)
- Setiap baris berlabel `data_type = "simulation"`.
- Hubungan target_value <-> RGB dibuat dengan formula linear + noise Gaussian
  yang SEPENUHNYA FIKTIF, hanya untuk menguji pipeline color-features -> model.
- Tidak berasal dari studi literatur atau pengukuran material apa pun.
- TIDAK BOLEH dikutip sebagai bukti hubungan warna-pH pada dressing nyata.

## Dataset eksperimen (belum ada)
Skema kolom yang disiapkan untuk data nyata (lihat FASE 5 master prompt):
`sample_id, date, material_id, pH_reference, image_path, R_mean, G_mean, B_mean,
H_mean, S_mean, V_mean, lighting_condition, smartphone_model, distance_cm, angle, replicate`

## Model kalibrasi (src/calibration.py)
4 model dibandingkan: Linear Regression, Polynomial Regression (derajat 2),
Random Forest (100 pohon), SVR (kernel RBF). Split train/test 80/20.
Model terbaik dipilih berdasarkan R2 tertinggi PADA DATA YANG DIMASUKKAN --
saat ini hanya data simulasi, sehingga pemilihan ini juga hanya berlaku
untuk data simulasi, bukan jaminan performa pada data nyata.

## Prinsip
Simulation Data ≠ Experimental Data. Setiap fungsi yang menghasilkan
metrik (MAE/RMSE/R2) menyertakan label dataset_type, dan UI menampilkan
peringatan eksplisit setiap kali angka tersebut berasal dari simulasi.
