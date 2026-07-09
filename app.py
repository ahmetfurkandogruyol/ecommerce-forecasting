import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ─────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────
st.set_page_config(
    page_title="E-Ticaret Analiz Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ─────────────────────────────────────────
# LSTM MODELİ TANIMLA
# ─────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ─────────────────────────────────────────
# VERİ YÜKLE
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    rfm = pd.read_csv('data/rfm_churn_data.csv', index_col=0)
    product_ts = pd.read_csv('data/product_ts.csv', index_col=0, parse_dates=True)
    return rfm, product_ts

@st.cache_resource
def load_model():
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    model = LSTMModel()
    model.load_state_dict(torch.load('models/lstm_model.pth', map_location='cpu'))
    model.eval()
    return model, scaler

rfm, product_ts = load_data()
model_lstm, scaler = load_model()

# ─────────────────────────────────────────
# BAŞLIK
# ─────────────────────────────────────────
st.title("🛒 E-Ticaret Analiz Dashboard")
st.markdown("**Online Retail II** veri seti üzerinde RFM Segmentasyonu, Churn Tahmini ve LSTM Satış Tahmini")
st.markdown("---")

# ─────────────────────────────────────────
# ÜST METRİKLER
# ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric("Toplam Müşteri", f"{len(rfm):,}")
col2.metric("Churn Oranı", f"%{rfm['Churn'].mean()*100:.1f}")
col3.metric("Ort. Harcama", f"£{rfm['Monetary'].mean():,.0f}")
col4.metric("Ort. Satın Alma", f"{rfm['Frequency'].mean():.1f}x")

st.markdown("---")

# ─────────────────────────────────────────
# SEKME 1: RFM SEGMENTASYONu
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 RFM Segmentasyonu", "🔴 Churn Tahmini", "📈 LSTM Satış Tahmini"])

with tab1:
    st.header("📊 RFM Müşteri Segmentasyonu")

    col1, col2 = st.columns(2)

    with col1:
        # Pasta grafik
        fig, ax = plt.subplots(figsize=(6, 5))
        segment_counts = rfm['Segment'].value_counts()
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        ax.pie(segment_counts.values,
               labels=segment_counts.index,
               autopct='%1.1f%%',
               colors=colors)
        ax.set_title('Musteri Segment Dagilimi')
        st.pyplot(fig)
        plt.close()

    with col2:
        # Segment profil tablosu
        st.subheader("Segment Profilleri")
        profile = rfm.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(1)
        profile.columns = ['Ort. Recency (gun)', 'Ort. Frequency', 'Ort. Monetary (GBP)']
        st.dataframe(profile, use_container_width=True)

        # Müşteri sayısı
        st.subheader("Segment Musteri Sayilari")
        count_df = rfm['Segment'].value_counts().reset_index()
        count_df.columns = ['Segment', 'Musteri Sayisi']
        st.dataframe(count_df, use_container_width=True)

    # Bar grafik
    st.subheader("Segmentlere Gore Ortalama Harcama")
    fig, ax = plt.subplots(figsize=(10, 4))
    segment_monetary = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
    ax.bar(segment_monetary.index, segment_monetary.values, color=colors)
    ax.set_ylabel('Ortalama Harcama (GBP)')
    ax.set_title('Segmentlere Gore Ortalama Harcama')
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────
# SEKME 2: CHURN TAHMİNİ
# ─────────────────────────────────────────
with tab2:
    st.header("🔴 Churn Tahmini")

    col1, col2 = st.columns(2)

    with col1:
        # Churn dağılımı
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_counts = rfm['Churn'].value_counts()
        ax.bar(['Aktif', 'Churn'], churn_counts.values, color=['#2ecc71', '#e74c3c'])
        ax.set_title('Aktif vs Churn Musteri')
        ax.set_ylabel('Musteri Sayisi')
        st.pyplot(fig)
        plt.close()

    with col2:
        # Churn olasılığı dağılımı
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(rfm['Churn_Probability'], bins=30, color='#3498db', edgecolor='white')
        ax.set_title('Churn Olasiligi Dagilimi')
        ax.set_xlabel('Churn Olasiligi')
        ax.set_ylabel('Musteri Sayisi')
        st.pyplot(fig)
        plt.close()

    # En yüksek churn riskli müşteriler
    st.subheader("🚨 En Yüksek Churn Riskli 20 Müşteri")
    high_risk = rfm.sort_values('Churn_Probability', ascending=False)[
        ['Frequency', 'Monetary', 'Segment', 'Churn_Probability']
    ].head(20).round(2)
    high_risk.columns = ['Satin Alma Sayisi', 'Toplam Harcama', 'Segment', 'Churn Riski']
    st.dataframe(high_risk, use_container_width=True)

# ─────────────────────────────────────────
# SEKME 3: LSTM TAHMİN
# ─────────────────────────────────────────
with tab3:
    st.header("📈 LSTM Satış Tahmini")
    st.markdown("**WHITE HANGING HEART T-LIGHT HOLDER** ürünü için LSTM tahmini")

    # Gerçek veri grafiği
    st.subheader("Geçmiş Satış Verisi")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(product_ts.index, product_ts['Daily_Quantity'], color='#3498db', linewidth=1)
    ax.set_title('Gunluk Satis Miktari')
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Satis Miktari')
    st.pyplot(fig)
    plt.close()

    # Gelecek tahmin
    st.subheader("Gelecek 30 Gün Tahmini")

    scaled_data = scaler.transform(product_ts[['Daily_Quantity']])
    last_sequence = scaled_data[-30:]
    last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0)

    future_preds = []
    current_seq = last_sequence.clone()

    model_lstm.eval()
    with torch.no_grad():
        for _ in range(30):
            pred = model_lstm(current_seq)
            future_preds.append(pred.item())
            new_val = pred.unsqueeze(0)
            current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    future_dates = pd.date_range(
        start=product_ts.index[-1] + pd.Timedelta(days=1),
        periods=30
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(product_ts.index[-60:], product_ts['Daily_Quantity'].values[-60:],
            label='Gecmis Satis', color='#3498db', linewidth=1.5)
    ax.plot(future_dates, future_preds,
            label='LSTM Tahmini (30 gun)', color='#e74c3c',
            linewidth=2, linestyle='--')
    ax.axvline(x=product_ts.index[-1], color='gray', linestyle=':', linewidth=1)
    ax.set_title('Gelecek 30 Gun Satis Tahmini')
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Satis Miktari')
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # Tahmin tablosu
    st.subheader("Tahmin Tablosu")
    pred_df = pd.DataFrame({
        'Tarih': future_dates.strftime('%Y-%m-%d'),
        'Tahmini Satis': future_preds.flatten().round(0).astype(int)
    })
    st.dataframe(pred_df, use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("**Geliştirici:** Ahmet Furkan | **Veri:** UCI Online Retail II | **Model:** LSTM + XGBoost + RFM")