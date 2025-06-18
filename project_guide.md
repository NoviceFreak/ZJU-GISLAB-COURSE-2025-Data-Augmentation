# Image-Editing Data Augmentation for Image Classification  
*Course Project Guide*

---

## 1. Introduction
Data augmentation broadens the diversity of training samples and helps vision models generalize better.  
In this project you will:

1. **Generate edited images** from an original image + text-prompt pair using an image-editing model.  
2. **Augment** a standard image-classification dataset with these synthetic images.  
3. **Compare** model performance when trained on  
   * real data only,  
   * synthetic data only, and  
   * the combination of both.

---

## 2. Prerequisites
| Requirement | Version / Notes |
|-------------|-----------------|
| **Python**  | ≥ 3.9 |
| **PyTorch** | ≥ 2.0 |
| **Hugging Face Transformers** | latest stable |
| **GPU**     | NVIDIA RTX 3090 / 4090 (available on school Linux servers) |

```bash
conda create -n img-edit-aug python=3.10
conda activate img-edit-aug
pip install torch torchvision transformers accelerate datasets
```

⸻

3. Repository Structure

```
├── data/                        # real, synthetic, mixed, test
├── checkpoints/                 # saved model weights
├── prompts.txt                  # list of text prompts
├── src/
│   ├── generate_images.py       # ⇢ create synthetic images
│   ├── train_model.py           # ⇢ train the classifier
│   ├── inference.py             # ⇢ run model on test set
│   └── evaluate.py              # ⇢ compute metrics
└── README.md                    # (this file)
```

⸻

4. Image Generation (src/generate_images.py)

Option A — Local pre-trained editor (provided by TAs)

```
python src/generate_images.py \
  --input_dir data/real \
  --prompts_file prompts.txt \
  --output_dir data/synthetic \
  --checkpoint checkpoints/editor_zeta.pt
```

Option B — Commercial image-editing API

Add your API key to api_keys.yaml, then run:

```
python src/generate_images.py \
  --input_dir data/real \
  --prompts_file prompts.txt \
  --output_dir data/synthetic \
  --provider commercial
```

The script writes edited images to data/synthetic/ using the same folder hierarchy as the originals.

⸻

5. Model Training (src/train_model.py)

Variant flag	Training data used
real	data/real/ only
synthetic	data/synthetic/ only
mixed	concatenation of data/real/ + data/synthetic/

Example (mixed data, ResNet-50):

```
python src/train_model.py \
  --variant mixed \
  --model resnet50 \
  --epochs 30 \
  --batch_size 64 \
  --save_path checkpoints/classifier_mixed.pt
```

⸻

6. Inference (src/inference.py)

```
python src/inference.py \
  --checkpoint checkpoints/classifier_mixed.pt \
  --test_dir data/test \
  --out predictions.json
```

⸻

7. Evaluation (src/evaluate.py)

```
python src/evaluate.py \
  --predictions predictions.json \
  --labels data/test/labels.json
```

Metrics reported: Top-1 accuracy, Top-5 accuracy, precision, recall, F1.

⸻

8. Experiments & Reporting

Design four complementary studies and record your findings in the tables provided below.
For each experiment, run three seeds (e.g. --seed 0 1 2) and report the mean ± std.

8.1 Data-Mix Ablation

Training Data	Top-1 Acc (%)	Top-5 Acc (%)
Real only		
Synthetic only		
Real + Synthetic (Mixed)		

Run the three --variant settings described in §5.

⸻

8.2 Pre-trained Model Size Scaling

Use a single architecture family (e.g. Vision Transformer) at different parameter counts.

Model (ViT)	Params (M)	Top-1 Acc (%)	Top-5 Acc (%)
vit_b16	~86 M		
vit_l16	~304 M		
vit_h14	~632 M		

Example command:

```
python src/train_model.py \
  --variant mixed \
  --model vit_b16 \
  --epochs 30 \
  --save_path checkpoints/vit_b16_mixed.pt
```

Repeat for vit_l16 and vit_h14.

⸻

8.3 Alternative Architecture Choice

Compare four architecture types with roughly similar parameter budgets.

Architecture	Params (M)	Top-1 Acc (%)	Top-5 Acc (%)
ResNet-50	26 M		
ConvNeXt-T	29 M		
EfficientNet-B4	19 M		
ViT-B/16	86 M		

Specify the architecture via --model (names as in timm or transformers).

⸻

8.4 Training Data-Samples Scaling

Hold the model fixed (pick your best from 8.3) and vary how much data you use.

Data Used	Real Images	Synthetic Images	Total	Top-1 Acc (%)	Top-5 Acc (%)
25 %					
50 %					
100 %					

Control sample size via --max_real N and --max_synth N flags that subsample the loaders.

