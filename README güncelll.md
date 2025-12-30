
# Yapay Zeka Destekli Görüntü Sınıflandırıcı

Bu proje, derin öğrenme teknikleri kullanılarak belirli kategorideki görselleri tanıyabilen bir yapay zeka uygulamasıdır. Proje kapsamında Vision Transformer (ViT) mimarisi kullanılarak bir görüntü sınıflandırma modeli eğitilmiş ve bu model kullanıcı dostu bir Gradio arayüzü ile entegre edilmiştir

# 📖 Proje Hakkında

Sistem, kullanıcıdan alınan görüntüleri otomatik olarak boyutlandırma ve normalizasyon gibi ön işleme adımlarından geçirir. Ardından, eğitilmiş model aracılığıyla görüntünün hangi sınıfa ait olduğunu tahmin ederek sonucu web arayüzünde metin olarak sunar.

Hedef: Makine öğrenimi tekniklerini kullanarak görsel verileri sınıflandırmak ve işlevsel bir uygulama üretmek.
Model: Hugging Face Vision Transformers (ViT).
Arayüz: Gradio.

# 🛠️ Kurulum ve Çalıştırma

1. Gereksinimlerin Yüklenmesi
Projenin çalışması için gerekli olan Python kütüphanelerini (torch, transformers, gradio vb.) aşağıdaki komutla yükleyebilirsiniz:
pip install -r requirements.txt

2. Uygulamanın Başlatılması
Görüntüleme arayüzünü başlatmak için ana dizinde şu komutu çalıştırın:
python app.py

# 🚀 Fonksiyonel Özellikler

Görüntü Yükleme: Kullanıcı bilgisayarından kolayca görsel yükleyebilir.
Ön İşleme: Görüntüler modelin beklediği 224x224 boyutuna otomatik getirilir ve normalize edilir.
Sınıflandırma: Yüklenen fotoğrafın hangi kategoriye ait olduğu anında kullanıcıya gösterilir.
Web Arayüzü: Yükleme butonu, fotoğraf ekranı ve tahmin sonuçlarını içeren modern bir tasarım sunulur.

# 📊 Eğitim Detayları ve Metrikler

Modelin eğitimi sırasında başarımı ölçmek için şu metrikler kullanılmıştır:
Metrik                      Açıklama
Accuracy (Doğruluk)         Modelin doğru tahmin yapma oranı.
Precision & Recall          Modelin sınıfları ayırt etme hassasiyeti.


# Proje Yapısı

trainingnazli.py: Modelin eğitim sürecini ve veri seti işlemlerini yöneten kodlar.
app.py: Gradio tabanlı kullanıcı arayüzü dosyası.
model_nazli/: Eğitilmiş modelin ağırlıkları ve konfigürasyon dosyaları.
requirements.txt: Proje için gerekli bağımlılıklar listesi.
README.md: Proje dokümantasyonu.

# 📊 Veri Seti (Dataset)
Bu projenin eğitim aşamasında aşağıdaki veri seti kullanılmıştır:

* **Veri Seti İsmi:** [Animals-10 Dataset:]
* **Kaynak:** [Veri Setine Gitmek İçin Tıklayın](https://www.kaggle.com/datasets/amankumar094/animal-dataset)

Eğitim veri seti üzerinden elde edilen model_nazli klasörü ve "sektör kampüste dersi ödevi" demo videosunun bulunduğu drive linki: https://drive.google.com/drive/u/0/my-drive

Eğitim sürecine ait Kayıp (Loss) ve Doğruluk (Accuracy) grafiklerine proje klasöründeki egitim_sonuclari.png dosyasından ulaşılabilir.

