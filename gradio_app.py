import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# LSTM Model
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

# Veri yükle
rfm = pd.read_csv('data/rfm_churn_data.csv', index_col=0)
product_ts = pd.read_csv('data/product_ts.csv', index_col=0, parse_dates=True)

# Model yükle
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
model_lstm = LSTMModel()
model_lstm.load_state_dict(torch.load('models/lstm_model.pth', map_location='cpu'))
model_lstm.eval()

# ─────────────────────────────────────────
# TAB 1: RFM SEGMENTASYONU
# ─────────────────────────────────────────
def rfm_segment_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pasta grafik
    segment_counts = rfm['Segment'].value_counts()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    axes[0].pie(segment_counts.values,
               labels=segment_counts.index,
               autopct='%1.1f%%',
               colors=colors)
    axes[0].set_title('Musteri Segment Dagilimi')
    
    # Bar grafik
    segment_monetary = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
    axes[1].bar(segment_monetary.index, segment_monetary.values, color=colors)
    axes[1].set_ylabel('Ortalama Harcama (GBP)')
    axes[1].set_title('Segmentlere Gore Ortalama Harcama')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# TAB 2: CHURN TAHMİNİ
# ─────────────────────────────────────────
def churn_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Churn dağılımı
    churn_counts = rfm['Churn'].value_counts()
    axes[0].bar(['Aktif', 'Churn'], churn_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Aktif vs Churn Musteri')
    axes[0].set_ylabel('Musteri Sayisi')
    
    # Churn olasılığı dağılımı
    axes[1].hist(rfm['Churn_Probability'], bins=30, color='#3498db', edgecolor='white')
    axes[1].set_title('Churn Olasiligi Dagilimi')
    axes[1].set_xlabel('Churn Olasiligi')
    axes[1].set_ylabel('Musteri Sayisi')
    
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# TAB 3: LSTM TAHMİNİ
# ─────────────────────────────────────────
def lstm_forecast():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(product_ts.index, product_ts['Daily_Quantity'], color='#3498db', linewidth=1)
    ax.set_title('WHITE HANGING HEART T-LIGHT HOLDER - Gunluk Satis')
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Satis Miktari')
    plt.tight_layout()
    return fig

def lstm_forecast_future():
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
    future_dates = pd.date_range(start=product_ts.index[-1] + pd.Timedelta(days=1), periods=30)

    fig, ax = plt.subplots(figsize=(12, 5))
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
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# GRADIO ARAYÜZÜ
# ─────────────────────────────────────────
with gr.Blocks(title="E-Ticaret Analiz Dashboard") as demo:
    gr.Markdown("# 🛒 E-Ticaret Analiz Dashboard")
    gr.Markdown("**Online Retail II** veri seti üzerinde RFM Segmentasyonu, Churn Tahmini ve LSTM Satış Tahmini")
    
    # Metrikler
    with gr.Row():
        gr.Metric(value=len(rfm), label="Toplam Müşteri")
        gr.Metric(value=f"{rfm['Churn'].mean()*100:.1f}%", label="Churn Oranı")
        gr.Metric(value=f"£{rfm['Monetary'].mean():,.0f}", label="Ortalama Harcama")
        gr.Metric(value=f"{rfm['Frequency'].mean():.1f}x", label="Ort. Satın Alma")
    
    with gr.Tabs():
        with gr.TabItem("📊 RFM Segmentasyonu"):
            gr.Plot(rfm_segment_analysis)
            
            with gr.Row():
                gr.Dataframe(rfm.groupby('Segment').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': 'mean'
                }).round(1))
        
        with gr.TabItem("🔴 Churn Tahmini"):
            gr.Plot(churn_analysis)
            
            gr.Markdown("### En Yüksek Churn Riskli 20 Müşteri")
            high_risk = rfm.sort_values('Churn_Probability', ascending=False)[
                ['Frequency', 'Monetary', 'Segment', 'Churn_Probability']
            ].head(20).round(2)
            high_risk.columns = ['Satın Alma', 'Harcama', 'Segment', 'Churn Riski']
            gr.Dataframe(high_risk)
        
        with gr.TabItem("📈 LSTM Satış Tahmini"):
            gr.Plot(lstm_forecast)
            gr.Plot(lstm_forecast_future)

if __name__ == "__main__":
    demo.launch()