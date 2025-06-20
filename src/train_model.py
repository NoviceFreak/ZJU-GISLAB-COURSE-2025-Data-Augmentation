import argparse
import os
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from transformers import AutoModel, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP/SigLIP models on image classification datasets")
    parser.add_argument("--variant", choices=["real", "synthetic", "mixed"], default="real", help="training data")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="model name on Hugging Face")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="checkpoints/model.pt")
    return parser.parse_args()


def build_dataloaders(variant: str, batch_size: int) -> (DataLoader, List[str]):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    def load(path):
        return datasets.ImageFolder(path, transform=transform)

    datasets_list = []
    root = Path("data")
    if variant in ["real", "mixed"]:
        datasets_list.append(load(root / "real"))
    if variant in ["synthetic", "mixed"]:
        datasets_list.append(load(root / "synthetic"))

    dataset = datasets_list[0] if len(datasets_list) == 1 else ConcatDataset(datasets_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader, dataset.classes


def compute_text_features(model, processor, class_names, device):
    prompts = [f"a photo of a {name}" for name in class_names]
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_text_features(**text_inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features


def train(model, processor, dataloader, class_features, device, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    logit_scale = model.logit_scale.exp().to(device)

    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            images = images.to(device)
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ class_features.T
            loss = criterion(logits, labels.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, class_names = build_dataloaders(args.variant, args.batch_size)
    model = AutoModel.from_pretrained(args.model).to(device)
    processor = AutoProcessor.from_pretrained(args.model)

    class_features = compute_text_features(model, processor, class_names, device)
    train(model, processor, dataloader, class_features, device, args.epochs)

    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
