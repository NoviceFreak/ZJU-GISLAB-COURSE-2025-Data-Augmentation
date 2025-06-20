import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets
from transformers import (
    AutoModel,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from datasets import Dataset as HFDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP/SigLIP models on image classification datasets"
    )
    parser.add_argument(
        "--variant",
        choices=["real", "synthetic", "mixed"],
        default="real",
        help="training data",
    )
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch16",
        help="model name on Hugging Face",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="checkpoints/model")
    return parser.parse_args()


def build_datasets(variant: str):
    def load(path):
        return datasets.ImageFolder(path)

    datasets_list = []
    root = Path("data")
    if variant in ["real", "mixed"]:
        datasets_list.append(load(root / "real"))
    if variant in ["synthetic", "mixed"]:
        datasets_list.append(load(root / "synthetic"))
    dataset = (
        datasets_list[0] if len(datasets_list) == 1 else ConcatDataset(datasets_list)
    )
    class_names = (
        datasets_list[0].classes
        if isinstance(dataset, datasets.ImageFolder)
        else datasets_list[0].classes
    )
    return dataset, class_names


class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.convert("RGB")
        processed = self.processor(images=img, return_tensors="pt")
        # Remove batch dimension
        item = {k: v.squeeze(0) for k, v in processed.items()}
        item["labels"] = label
        return item


class CLIPForClassification(torch.nn.Module):
    def __init__(self, model, class_features):
        super().__init__()
        self.model = model
        self.class_features = class_features
        self.logit_scale = getattr(model, "logit_scale", None)

    def forward(self, **inputs):
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scale = self.logit_scale.exp() if self.logit_scale is not None else 1.0
        logits = scale * image_features @ self.class_features.T
        return {"logits": logits}


def compute_text_features(model, processor, class_names, device):
    prompts = [f"a photo of a {name}" for name in class_names]
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_text_features(**text_inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, class_names = build_datasets(args.variant)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    class_features = compute_text_features(model, processor, class_names, device)
    class_features = class_features.to(device)

    train_dataset = ImageClassificationDataset(dataset, processor)

    model_wrapper = CLIPForClassification(model, class_features)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(args.save_path)
    processor.save_pretrained(args.save_path)
    print(f"Model and processor saved to {args.save_path}")


if __name__ == "__main__":
    main()
