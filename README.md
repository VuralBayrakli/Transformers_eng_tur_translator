
# Transformers kullanılarak İngilizce Türkçe Çeviri Uygulaması

Bu projede Transformers kullanılarak Türkçe İngilizce çevirisi yapabilen uygulama geliştirilmiştir.

## Dosya Yükleme

Projeyi `git` ile yükleyin

```bash
    https://github.com/VuralBayrakli/Transformers_eng_tur_translator.git
```

Modeli indirin

    ### https://github.com/VuralBayrakli/Transformers_eng_tur_translator/raw/master/transformer_en_tr_son2.pt?download=


Ana Klasörüne Giriş Yapın

Sanal ortam kurun

```bash
    python -m venv venvtranslator
```

Sanal ortamı aktive edin

```bash
    venvtranslator\Scripts\activate
```

Gerekli kütüphanelerin yükleyin
```bash
    pip install -r requirements.txt
```

Projeyi çalıştırın
```bash
    python tahmin.py
```

## Veri setinden birkaç örnek

![App Screenshot](https://github.com/VuralBayrakli/Transformers_eng_tur_translator/blob/master/screenshots/ss1.png)

## Veri setinde en çok kullanılan kelimeler

![App Screenshot](https://github.com/VuralBayrakli/Transformers_eng_tur_translator/blob/master/screenshots/ss4.png)

## Tahmin gerçekleştirme

![App Screenshot](https://github.com/VuralBayrakli/Transformers_eng_tur_translator/blob/master/screenshots/ss6.png)
