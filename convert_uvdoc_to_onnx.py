# Gerekli Kütüphaneler
import torch
import onnx
from utils import load_model
IMG_SIZE = [488, 712]

# UVDoc Modelini yükleme fonksiyonu
def load_uvdoc_model(model_path='model/best_model.pkl'):
    model = load_model(model_path)
    model.eval()
    return model

# Modeli ONNX formatına dönüştüren fonksiyon
def convert_to_onnx(model, output_path='model/uvdoc_model.onnx', img_size=IMG_SIZE):
    # Sahte girdi oluştur (örneğin 3 kanallı RGB görüntü için)
    dummy_input = torch.randn(1, 3, img_size[1], img_size[0])

    # Modeli ONNX'e dönüştür
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True
    )
    print(f"Model başarıyla '{output_path}' olarak ONNX formatına dönüştürüldü.")

# ONNX modeli doğrulama fonksiyonu
def validate_onnx_model(onnx_model_path='uvdoc_model.onnx'):
    try:
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model doğrulandı. Model geçerli.")
    except Exception as e:
        print(f"ONNX model doğrulanırken hata oluştu: {e}")

# Ana işlem
if __name__ == "__main__":
    # Modeli yükle
    model = load_uvdoc_model()

    # ONNX'e dönüştür
    convert_to_onnx(model)

    # Dönüştürülen modeli doğrula
    validate_onnx_model()
