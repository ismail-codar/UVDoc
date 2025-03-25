import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort

IMG_SIZE = [488, 712]

def bilinear_unwarping(warped_img, point_positions, img_size):
    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True)
    return unwarped_img

def unwarp_img_onnx(onnx_model_path, img_path, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ONNX modeli yükle
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    # Resmi yükle ve ön işlemler
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    resized_img = cv2.resize(img_rgb, img_size).transpose(2, 0, 1)
    inp = resized_img[np.newaxis, :, :, :].astype(np.float32)

    # ONNX ile tahmin yap
    outputs = session.run(None, {'input': inp})
    point_positions2D = outputs[0]

    # Unwarp işlemi (PyTorch tensor kullanarak devam)
    size = img_rgb.shape[:2][::-1]
    unwarped = bilinear_unwarping(
        warped_img=torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device),
        point_positions=torch.from_numpy(point_positions2D).to(device),
        img_size=size,
    )

    unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Sonucu kaydet
    unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.splitext(img_path)[0] + "_unwarp.png", unwarped_BGR)

    print(f"Unwarped image saved as {os.path.splitext(img_path)[0]}_unwarp.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-path", type=str, default="./model/uvdoc_model.onnx", help="Path to the ONNX model."
    )
    parser.add_argument("--img-path", type=str, help="Path to the document image to unwarp.")

    args = parser.parse_args()
    unwarp_img_onnx(args.onnx_path, args.img_path, IMG_SIZE)
