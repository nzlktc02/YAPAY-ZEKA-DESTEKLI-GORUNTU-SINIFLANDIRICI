import gradio as gr
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os

# --- 1. MODEL YOLU ---
MODEL_PATH = "./model_nazli" 

try:
    print(f"🔄 Model yükleniyor: {MODEL_PATH}")
    # Eğittiğin ViT modelini ve işlemcisini yüklüyoruz 
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model.eval() 
    
    # --- KRİTİK DÜZELTME ---
    if hasattr(model.config, "id2label") and model.config.id2label:
        # Anahtarları string yaparak KeyError hatasını engelliyoruz
        model.config.id2label = {str(k): v for k, v in model.config.id2label.items()}
        print(f"✅ Etiketler yüklendi: {model.config.id2label}")
    
    print("🚀 Sistem hazır!")
except Exception as e:
    print(f"❌ Yükleme Hatası: {e}")

# --- 2. TAHMİN FONKSİYONU ---
def predict_image(img):
    if img is None: 
        return "Lütfen bir resim yükleyin."
    
    try:
        # Görüntü ön işleme
        inputs = processor(images=img.convert("RGB"), return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Tahmin edilen sınıf indeksi
        predicted_class_idx = logits.argmax(-1).item()
        idx_str = str(predicted_class_idx)
        
        # Etiketi bulmaya çalış
        if idx_str in model.config.id2label:
            label = model.config.id2label[idx_str]
        else:
            label = f"Tanımlanamayan Nesne (ID: {idx_str})"
        
        # Olasılık skorunu hesapla
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][predicted_class_idx].item()
        
        return {label: float(confidence)}

    except Exception as e:
        return f"Tahmin sırasında hata oluştu: {str(e)}"

# --- 3. ARAYÜZ TASARIMI (Gradio 6.0+ Düzgün Diziliş) ---
with gr.Blocks(title="AI Görüntü Sınıflandırıcı") as demo:
    gr.Markdown("# 🖼️ Yapay Zeka Görüntü Sınıflandırıcı")
    gr.Markdown("Eğitilmiş modelinizi test etmek için bir görsel yükleyin.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Görsel Seç")
            predict_btn = gr.Button("🔍 Tahmin Et", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Tahmin Sonucu")
    
    # Buton tetikleyicisi
    predict_btn.click(fn=predict_image, inputs=image_input, outputs=output_label)

# --- 4. BAŞLATMA ---
if __name__ == "__main__":
    # Gradio 6.0 kuralı: Temayı Blocks içinde değil, launch içinde tanımlıyoruz.
    demo.launch(theme=gr.themes.Soft())