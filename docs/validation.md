# Validation

Status: infrastruktur pengujian sudah lengkap (src/validation.py + halaman
"Experimental Data" di UI), tapi BELUM ADA hasil aktual karena belum ada
data eksperimen nyata dari smart dressing.

## Fungsi yang tersedia (src/validation.py)
| Fungsi | Menguji |
|---|---|
| repeatability_test | Konsistensi RGB/HSV pada sampel yang sama, dikelompokkan per sample_id + kondisi pengambilan gambar |
| lighting_variation_test | Variasi warna antar kondisi pencahayaan |
| smartphone_variation_test | Variasi warna antar model smartphone |
| distance_variation_test | Variasi warna antar jarak kamera |
| angle_variation_test | Variasi warna antar sudut kamera |

Semua fungsi murni menghitung mean/std/count dari data yang diberikan --
tidak ada angka yang dihardcode atau diasumsikan. Begitu data eksperimen
diunggah lewat halaman "Experimental Data", hasil pengujian ini dihitung
langsung dari data tersebut secara real-time.

## Metrik kalibrasi (MAE/RMSE/R2)
Halaman "Experimental Data" bisa melatih 4 model (Linear/Polynomial/RF/SVR)
dari data eksperimen begitu tersedia. Metriknya dihitung sungguhan dari
data yang diunggah -- tapi TIDAK ADA data eksperimen di repo ini, jadi
belum ada angka yang bisa dilaporkan.
