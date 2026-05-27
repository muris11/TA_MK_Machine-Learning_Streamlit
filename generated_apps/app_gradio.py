from pathlib import Path
import sys
import pandas as pd
import gradio as gr

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from prediction_service import predict_condition


def run_prediction(tahun, gini, tpt, inflasi, ipm):
    result = predict_condition(tahun, gini, tpt, inflasi, ipm)
    actions = "\n".join([f"- {item}" for item in result["aksi_kebijakan"]])
    timeline = "\n".join([f"- **{period}**: {'; '.join(items)}" for period, items in result["timeline"].items()])
    markdown = (
        "### Hasil Analisis Kondisi Sosial\n\n"
        f"- **Tahun target:** {int(tahun)}\n"
        f"- **Estimasi angka kemiskinan:** {result['prediksi_kemiskinan']:.2f}%\n"
        f"- **Level prioritas intervensi:** {result['priority_level']}\n"
        f"- **Status:** {result['status']}\n\n"
        "### Rekomendasi Utama\n\n"
        f"{result['rekomendasi_utama']}\n\n"
        "### Aksi Kebijakan yang Disarankan\n\n"
        f"{actions}\n\n"
        "### Timeline Implementasi\n\n"
        f"{timeline}"
    )
    table = pd.DataFrame([
        {"Periode": period, "Aksi": "; ".join(items)} for period, items in result["timeline"].items()
    ])
    return markdown, table

with gr.Blocks(title="Dashboard Prediksi Kondisi Sosial Jawa Barat") as demo:
    gr.Markdown("# Dashboard Prediksi Kondisi Sosial Jawa Barat dan Rekomendasi Kebijakan")
    gr.Markdown("Sistem prediksi dan dukungan keputusan berbasis model machine learning.")
    with gr.Row():
        with gr.Column():
            tahun = gr.Number(label="Tahun Prediksi", value=2029)
            gini = gr.Number(label="Gini Ratio", value=400)
            tpt = gr.Number(label="Tingkat Pengangguran Terbuka (%)", value=5.0)
            inflasi = gr.Number(label="Rata-rata Inflasi Tahunan (%)", value=0.15)
            ipm = gr.Number(label="Indeks Pembangunan Manusia", value=73.5)
            button = gr.Button("Jalankan Prediksi", variant="primary")
        with gr.Column():
            narrative = gr.Markdown()
            timeline = gr.Dataframe(label="Timeline Kebijakan")
    button.click(run_prediction, inputs=[tahun, gini, tpt, inflasi, ipm], outputs=[narrative, timeline])

if __name__ == "__main__":
    demo.launch()
