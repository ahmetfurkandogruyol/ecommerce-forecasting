# 🛒 E-Ticaret RFM & Churn & LSTM Analiz

UCI Online Retail II veri seti üzerinde müşteri segmentasyonu, churn tahmini ve satış tahmini.

## 📊 Projede Neler Var?

### 1. RFM Müşteri Segmentasyonu
- 4,338 müşteri 4 segmente ayrıldı
- Segmentler: Şampiyonlar, Sadık Müşteriler, Risk Altında, Kayıp Müşteriler
- Her segmentin profili analiz edildi

### 2. Churn Tahmini (XGBoost)
- %70 doğrulukla müşteri kaybını tahmin
- Frequency en önemli özellik
- En yüksek risk grubunu tespit et

### 3. LSTM Satış Tahmini (PyTorch)
- WHITE HANGING HEART T-LIGHT HOLDER ürünü seçildi
- Geçmiş 30 gün verisi → Gelecek 30 gün tahmini
- MAE: 74 adet (ortalama hatası)

## 🚀 Nasıl Çalıştırılır?

### Gereksinimler
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install torch xgboost lightgbm
pip install jupyter streamlit kaggle plotly
```

### Veri İndir
```bash
kaggle datasets download -d ulrikthygepsen/online-retail-ii
unzip online-retail-ii.zip -d data/
```

### Analiz Çalıştır
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Dashboard Çalıştır
```bash
streamlit run app.py
```

Tarayıcıda açılacak: `http://localhost:8501`

## 📁 Klasör Yapısı

## 📊 Veri Seti

- **Online Retail II** (UCI Machine Learning Repository)
- 2010-2011 İngiltere e-ticaret şirketi
- 541,910 satır → 397,885 temiz satır
- 4,338 müşteri, 4,070 ürün

## 🔍 Temel Bulgular

| Metrik | Değer |
|---|---|
| Toplam Müşteri | 4,338 |
| Churn Oranı | %39.49 |
| Ortalama Harcama | £2,054 |
| Ort. Satın Alma | 4.27x |
| Şampiyon Müşteri | %22 |

## 🛠️ Kullanılan Teknolojiler

- **Veri İşleme:** Pandas, NumPy, Scikit-learn
- **Derin Öğrenme:** PyTorch LSTM
- **Makine Öğrenmesi:** XGBoost, LightGBM
- **Görselleştirme:** Matplotlib, Seaborn, Plotly, Streamlit
- **Versiyon Kontrol:** Git, GitHub
- **Deploy:** Hugging Face Spaces

## 📈 Model Performansı

### Churn Tahmini (XGBoost)
- Accuracy: %70
- Önemli Özellik: Frequency (Satın Alma Sıklığı)

### LSTM Satış Tahmini
- MAE: 74 adet
- RMSE: 101 adet
- Trend Yakalama: ✅

## 🎯 İş Soruları ve Cevapları

**Q: Hangi müşteri segmenti en değerli?**
- A: Şampiyonlar (Ortalama £6,358 harcama)

**Q: Churn riski olan müşteriler kimler?**
- A: Sık satın almayan (Frequency < 3), uzun süre alışveriş yapmayan (Recency > 120)

**Q: Ürün satışını nasıl tahmin ederiz?**
- A: LSTM ile geçmiş 30 gün → gelecek 30 gün

## 👤 Geliştirici

**Ahmet Furkan** - Veri Bilimci & Makine Öğrenmesi Mühendisi

LinkedIn: [linkedin.com/in/ahmet-furkan](#)

## 📄 Lisans

MIT License