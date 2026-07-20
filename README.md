# Indonesian License Plate OCR using Visual Language Model (LM Studio)

## Deskripsi

Proyek ini merupakan implementasi **Optical Character Recognition (OCR)** pada plat nomor kendaraan Indonesia menggunakan **Visual Language Model (VLM)** yang dijalankan melalui **LM Studio** dan diintegrasikan menggunakan **Python**.

Model VLM menerima gambar plat nomor sebagai input, kemudian menghasilkan prediksi nomor plat. Hasil prediksi dibandingkan dengan ground truth menggunakan metrik **Character Error Rate (CER)**.

---

## Dataset

Dataset yang digunakan:

**Indonesian License Plate Dataset**

https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset

Dataset yang digunakan adalah **Indonesian License Plate Recognition Dataset**.

---

## Model

Visual Language Model yang digunakan:

- LLaVA 1.6 Mistral 7B

Model dijalankan menggunakan **LM Studio** melalui OpenAI Compatible API.

---

## Requirement

- Python 3.10+
- LM Studio
- Model LLaVA 1.6 Mistral 7B

Install dependency:

```bash
pip install -r requirements.txt
```

atau

```bash
pip install openai pandas jiwer tqdm
```

---

## Struktur Folder

```
project/
│
├── Indonesian License Plate Recognition Dataset/
│   ├── images/
│   ├── labels/
│   ├── train.txt
│   ├── val.txt
│   └── classes.names
│
├── main.py
├── ocr_results.csv
├── requirements.txt
└── README.md
```

---

## Menjalankan LM Studio

1. Buka LM Studio.
2. Download model **LLaVA 1.6 Mistral 7B**.
3. Masuk ke menu **Developer**.
4. Jalankan **OpenAI Compatible Server**.

Pastikan server berjalan pada:

```
http://localhost:1234/v1
```

---

## Menjalankan Program

Jalankan program menggunakan:

```bash
python main.py
```

Program akan:

1. Membaca seluruh gambar pada dataset.
2. Mengirim gambar ke LM Studio.
3. Melakukan OCR menggunakan Visual Language Model.
4. Membaca ground truth dari label dataset.
5. Menghitung Character Error Rate (CER).
6. Menyimpan hasil evaluasi ke file CSV.

---

## Output

Program menghasilkan file:

```
ocr_results.csv
```

dengan format:

| image | ground_truth | prediction | CER_score |
|--------|--------------|------------|-----------|
| test001.jpg | B9140BCD | B9140BCD | 0.0000 |
| test002.jpg | BG1352AE | BCG13562AE | 0.2500 |

---

## Character Error Rate (CER)

Evaluasi dilakukan menggunakan **Character Error Rate (CER)**.

Rumus:

```
CER = (S + D + I) / N
```

dengan:

- **S** = jumlah karakter yang mengalami substitusi
- **D** = jumlah karakter yang dihapus (Deletion)
- **I** = jumlah karakter yang disisipkan (Insertion)
- **N** = jumlah karakter pada ground truth

Semakin kecil nilai CER, semakin baik hasil OCR.

Nilai:

- CER = 0 → Prediksi sama dengan ground truth.
- CER mendekati 0 → Prediksi sangat baik.
- CER besar → Prediksi memiliki banyak kesalahan karakter.

---

## Cara Kerja

1. Membaca gambar plat nomor.
2. Mengubah gambar menjadi Base64.
3. Mengirim gambar ke LM Studio menggunakan OpenAI Compatible API.
4. Memberikan prompt agar model hanya mengembalikan nomor plat.
5. Membersihkan hasil prediksi menggunakan Regular Expression.
6. Membaca ground truth dari anotasi YOLO OCR.
7. Menghitung CER.
8. Menyimpan seluruh hasil ke file CSV.

---

## Hasil

Output program berupa:

- Prediksi nomor plat
- Ground truth
- Nilai Character Error Rate (CER)
- Nilai rata-rata CER seluruh dataset

---

## Author

Nama: **Muhammad Garan Renazi**

Teknik Robotika

Politeknik Negeri Batam
