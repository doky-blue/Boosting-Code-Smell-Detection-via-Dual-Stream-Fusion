# Boosting Code Smell Detection via Dual-Stream Fusion

This repository contains the implementation of a novel dual-stream multi-modal fusion approach for automated code smell detection. Our method synergistically combines deep textual semantics via LoRA-tuned CodeBERT with complex topological structures via Graph Attention Networks (GAT) operating on Code Property Graphs (CPG).

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Data Preprocessing](#data-preprocessing)
  - [Training CodeBERT Model](#training-codebert-model)
  - [Training GNN Model](#training-gnn-model)
  - [Model Fusion and Evaluation](#model-fusion-and-evaluation)
- [Scripts and Configuration](#scripts-and-configuration)
- [Hardware Requirements](#hardware-requirements)
- [Key Features](#key-features)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

Code smells serve as latent indicators of design and implementation flaws during software evolution, significantly impeding software maintainability and sustainability. This project proposes a novel dual-stream multi-modal fusion approach that addresses the limitations of existing single-perspective feature extraction methods.

### Key Innovations

1. **Dual-Stream Fusion Architecture**: Simultaneously captures textual semantics and structural dependencies
2. **LoRA-Fine-Tuned CodeBERT**: Parameter-efficient fine-tuning for semantic feature extraction
3. **GAT-JK on CPG**: Multi-scale structural feature extraction with jumping knowledge mechanism
4. **Focal Loss Mechanism**: Mitigates extreme class imbalance in code smell detection
5. **Stacking-LR Meta-Learner**: Decision-level fusion strategy for heterogeneous modalities

## 🏗️ Architecture

The framework consists of two parallel processing pipelines:

```
┌─────────────────┐     ┌──────────────────────────────┐
│  Source Code    │────▶│  Data Preprocessing           │
└─────────────────┘     │  - CPG Construction (Joern)   │
                        │  - AST Parsing (tree-sitter)  │
                        └──────────┬───────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐
    │  Stream 1: CodeBERT         │   │  Stream 2: GAT-JK          │
    │  - Tokenization            │   │  - CPG Graph Construction  │
    │  - LoRA Fine-tuning       │   │  - Multi-scale Attention   │
    │  - Semantic Features       │   │  - Structural Features     │
    └─────────────┬─────────────┘   └─────────────┬─────────────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 ▼
                  ┌─────────────────────────────┐
                  │  Stacking-LR Meta-Learner   │
                  │  - Logistic Regression      │
                  │  - Decision Fusion          │
                  └──────────────┬──────────────┘
                                 ▼
                  ┌─────────────────────────────┐
                  │  Final Classification       │
                  │  - Blob, Data Class         │
                  │  - Feature Envy, Long Method│
                  └─────────────────────────────┘
```

## 📊 Dataset

### Dataset Description

The project uses the MLCQ (Multi-Label Code Quality) benchmark dataset, which contains Java source code samples annotated with four types of code smells:

| Smell Type | Description | Characteristics |
|------------|-------------|-----------------|
| **Blob** | God Class / Data Hoarder | Large class with excessive responsibilities |
| **Data Class** | Data Container | Class with only fields and accessors |
| **Feature Envy** | misplaced responsibility | Methods excessively using another class |
| **Long Method** | Complex Function | Methods exceeding typical complexity thresholds |

### Dataset Statistics

- **Total Samples**: ~6,000+ Java code snippets
- **Class Distribution**: Highly imbalanced (77%+ samples are smell-free)
- **Data Format**: JSONL (JSON Lines) with code snippets and labels

### Dataset Directory Structure

```bash
dataset/
├── MLCQCodeSmellSamples.csv       # Original dataset
├── MLCQCodeSmellSamples.xlsx       # Spreadsheet format
├── process/
│   ├── tree-sitter-java/          # Java AST parser bindings
│   ├── get_four_model_data.py    # Data filtering for each smell type
│   ├── merge_and_split_jsonl.py   # Dataset splitting script
│   ├── get_jsonl.py               # JSONL format conversion
│   └── unders_over_sampling.py   # Sampling strategies
└── source.xlsx                     # Source data
```

## 🔧 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU acceleration)
- **System Dependencies**: Java Development Kit (for Joern analysis)

### 1. Create Python Environment

```bash
# Create a new conda environment
conda create -n code_smell python=3.10

# Activate the environment
conda activate code_smell

# Or use virtualenv
python -m venv code_smell_env
source code_smell_env/bin/activate  # Linux/Mac
code_smell_env\Scripts\activate     # Windows
```

### 2. Install Core Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related libraries
pip install transformers==4.35.0
pip install datasets==2.14.0
pip install accelerate==0.24.0

# Install PEFT for LoRA
pip install peft==0.6.0

# Install graph neural network libraries
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install data processing libraries
pip install pandas==2.1.0
pip install numpy==1.24.0
pip install scikit-learn==1.3.0

# Install XGBoost for ensemble methods
pip install xgboost==2.0.0

# Install tree-sitter for Java parsing
pip install tree-sitter==0.20.0
```

### 3. Install Tree-Sitter Java Parser

```bash
cd dataset/process/tree-sitter-java
pip install -e .
```

### 4. Install Joern (for CPG Generation)

```bash
cd joern
chmod +x joern-install.sh
./joern-install.sh
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import torch_geometric; print(f'Torch Geometric version: {torch_geometric.__version__}')"
```

## 📁 Project Structure

```
Boosting-Code-Smell-Detection-via-Dual-Stream-Fusion/
├── README.md
├── model/                          # CodeBERT-based models
│   ├── main.py                    # Main training script
│   ├── run.sh                     # Training launch script
│   ├── eval.py                    # Evaluation utilities
│   ├── predict.py                 # Prediction utilities
│   ├── stacking.py                # Stacking ensemble
│   ├── merge_and_perdict.py       # Result merging
│   ├── Focal_Loss_main.py         # Focal loss implementation
│   ├── focal_run.sh               # Focal loss training script
│   └── gnn/                       # GNN integration
│       ├── train_graphformer.py  # GNN training
│       ├── dataset.py            # Graph dataset loader
│       └── vocab.json             # Node type vocabulary
├── ablation/                       # Ablation study modules
│   ├── codebert/                 # CodeBERT ablation
│   │   ├── main.py
│   │   ├── eval.py
│   │   ├── predict.py
│   │   └── tune_hyperparameters.py
│   └── gnn/                      # GNN ablation
│       ├── train_gnn.py
│       ├── train_focal_loss.py
│       ├── cpg_dataset.py
│       └── test.py
├── fusion_evaluation/            # Model fusion experiments
│   ├── stacking_merge.py         # Stacking ensemble fusion
│   ├── stacking_XGBoost.py       # XGBoost-based fusion
│   └── grid_merge.py            # Grid search fusion
├── evaluation/                   # Baseline model evaluation
│   ├── SVM/                      # SVM baseline
│   │   ├── main.py
│   │   └── svm.py
│   └── random_forest/            # Random Forest baseline
│       ├── main.py
│       ├── eval.py
│       ├── predict.py
│       └── rf.py
├── loss_train/                   # Loss function experiments
│   ├── codebert/                 # Focal loss for CodeBERT
│   │   ├── main.py
│   │   ├── Focal_Loss_main.py
│   │   └── focal_run.sh
│   └── gnn/                      # Focal loss for GNN
│       ├── train_focal_loss.py
│       └── train_gnn.py
├── dataset/                      # Dataset and preprocessing
│   ├── process/                  # Data processing scripts
│   │   ├── get_four_model_data.py
│   │   ├── merge_and_split_jsonl.py
│   │   ├── get_jsonl.py
│   │   ├── get_link_and_name.py
│   │   └── unders_over_sampling.py
│   └── MLCQCodeSmellSamples.csv  # Dataset file
├── joern/                        # CPG generation tools
│   ├── joern-install.sh          # Installation script
│   └── prepare_java.py           # Java file preparation
└── temp.py                       # Utility scripts
```

## 🚀 Quick Start

### Data Preprocessing

#### Step 1: Download and Prepare Dataset

```bash
# Place your MLCQ dataset file in the dataset directory
cp MLCQCodeSmellSamples.csv dataset/

# Run data preprocessing
cd dataset/process
python get_link_and_name.py        # Extract links and names from dataset
python get_jsonl.py                # Convert to JSONL format
```

#### Step 2: Filter Data for Each Smell Type

```bash
# Generate separate datasets for each code smell
python get_four_model_data.py

# This creates:
# - blob_train.jsonl, blob_eval.jsonl, blob_test.jsonl
# - data_class_train.jsonl, data_class_eval.jsonl, data_class_test.jsonl
# - feature_envy_train.jsonl, feature_envy_eval.jsonl, feature_envy_test.jsonl
# - long_method_train.jsonl, long_method_eval.jsonl, long_method_test.jsonl
```

#### Step 3: Merge and Split Dataset

```bash
# Merge datasets and split into train/eval/test
python merge_and_split_jsonl.py

# Output format: {smell_type}_{train|eval|test}_four_model.jsonl
```

### Training CodeBERT Model

#### Configuration

Edit `model/run.sh` with your paths:

```bash
python main.py \
    --type "feature_envy" \                    # smell type: blob, data_class, feature_envy, long_method
    --output_dir "./output" \
    --train_data_file "/path/to/your/data/train.jsonl" \
    --eval_data_file "/path/to/your/data/eval.jsonl" \
    --test_data_file "/path/to/your/data/test.jsonl" \
    --model_name_or_path "/path/to/codebert-base/" \
    --do_train \
    --do_test \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --num_train_epochs 100
```

#### Run Training

```bash
cd model
bash run.sh

# Or with focal loss
bash focal_run.sh
```

#### Training Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `--type` | Code smell type | blob/data_class/feature_envy/long_method |
| `--train_batch_size` | Training batch size | 16 (GPU: 32) |
| `--eval_batch_size` | Evaluation batch size | 8 (GPU: 16) |
| `--learning_rate` | Learning rate | 5e-5 |
| `--num_train_epochs` | Number of epochs | 100 |
| `--max_grad_norm` | Gradient clipping | 1.0 |

### Training GNN Model

#### Step 1: Generate Code Property Graphs (CPG)

```bash
# Install Joern and generate CPGs
cd joern
bash joern-install.sh

# Prepare Java files for CPG generation
python prepare_java.py
```

#### Step 2: Convert CPGs to Graph Format

```bash
cd ablation/GNN

# Parse Java files and generate CPG graphs
bash parse_java_files.sh

# Convert to PyTorch Geometric format
python process_snippets.py
```

#### Step 3: Train GAT-JK Model

```bash
cd ablation/GNN

# Standard training
python train_gnn.py

# With focal loss for class imbalance
python train_focal_loss.py
```

### Model Fusion and Evaluation

#### Stacking with Logistic Regression

```bash
cd fusion_evaluation

# Run stacking fusion
python stacking_merge.py
```

#### XGBoost-based Fusion

```bash
cd fusion_evaluation

# Run XGBoost fusion
python stacking_XGBoost.py
```

#### Grid Search Fusion

```bash
cd fusion_evaluation

# Run grid search for optimal weights
python grid_merge.py
```

## 📝 Scripts and Configuration

### Data Processing Scripts

| Script | Purpose |
|--------|---------|
| `get_four_model_data.py` | Filter dataset for each smell type |
| `merge_and_split_jsonl.py` | Merge and split datasets |
| `get_jsonl.py` | Convert CSV to JSONL format |
| `get_link_and_name.py` | Extract links and names |
| `unders_over_sampling.py` | Handle class imbalance |

### Model Training Scripts

| Script | Purpose |
|--------|---------|
| `model/main.py` | CodeBERT training with LoRA |
| `model/Focal_Loss_main.py` | CodeBERT with focal loss |
| `ablation/GNN/train_gnn.py` | GAT-JK training |
| `ablation/GNN/train_focal_loss.py` | GAT with focal loss |

### Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `model/eval.py` | Standard evaluation metrics |
| `fusion_evaluation/stacking_merge.py` | Stacking ensemble evaluation |
| `evaluation/SVM/main.py` | SVM baseline evaluation |
| `evaluation/random_forest/main.py` | Random Forest baseline |

## 💻 Hardware Requirements

### Minimum Requirements

- **CPU**: Intel Core i7 or equivalent
- **RAM**: 16 GB
- **Storage**: 50 GB free space
- **GPU**: Not required for inference (but recommended for training)

### Recommended Requirements

- **CPU**: Intel Xeon or AMD EPYC
- **RAM**: 32 GB
- **Storage**: 100 GB SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080 or better)

### Software Dependencies

- **CUDA**: 11.0 or higher
- **cuDNN**: 8.0 or higher
- **Java**: JDK 11 or higher (for Joern)
- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11

## ✨ Key Features

### 1. LoRA Fine-tuning

Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning of CodeBERT:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

model = get_peft_model(model, lora_config)
```

### 2. Focal Loss for Class Imbalance

Binary focal loss handles the severe class imbalance in code smell datasets:

```python
class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt).pow(self.gamma)
        loss = focal_term * bce_loss
        return loss.mean()
```

### 3. GAT-JK Architecture

Graph Attention Network with Jumping Knowledge for multi-scale structural features:

```python
class GAT_JK_Pool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.jk = JumpingKnowledge()
        self.classifier = torch.nn.Linear(hidden_channels * num_heads * 2, out_channels)
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
