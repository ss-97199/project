"""
Inference on a local directory of images.

Usage:
    python infer.py --checkpoint runs/best_model.pth \
                    --input_dir /path/to/images_for_inference \
                    --output_csv preds.csv
"""

import argparse
import csv
import os
from typing import List

import torch
from PIL import Image
import torchvision.transforms as T

from config import IMAGE_SIZE
from model import build_model
from utils import load_checkpoint


def load_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def list_images(input_dir: str) -> List[str]:
    paths = []
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ):
            paths.append(os.path.join(input_dir, fname))
    return sorted(paths)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=False).to(device)
    model = load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    transform = load_transforms()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    img_paths = list_images(args.input_dir)
    if not img_paths:
        print(f"No images found in {args.input_dir}")
        return

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_label", "prob_class0", "prob_class1"])

        with torch.no_grad():
            for img_path in img_paths:
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())

                writer.writerow(
                    [
                        os.path.basename(img_path),
                        pred,
                        float(probs[0]),
                        float(probs[1]),
                    ]
                )

                print(f"{os.path.basename(img_path)} -> class {pred}, probs={probs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory of images for inference",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Path to save CSV predictions",
    )
    args = parser.parse_args()
    main(args)