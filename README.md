# Federated Learning PyTorch

A modular and extensible federated learning framework supporting multiple algorithms:

- **FedAvg**: Federated Averaging (baseline)
- **FedLoRA**: Federated Low-Rank Adaptation (parameter-efficient)
- **FedSDG**: Federated Structure-Decoupled Gating (dual-path personalization)

## 📁 Project Structure

```
Federated-Learning-PyTorch/
├── fl/                           # Core federated learning library
│   ├── algorithms/               # Aggregation algorithms
│   │   ├── fedavg.py            # FedAvg aggregation
│   │   └── fedlora.py           # FedLoRA/FedSDG aggregation
│   ├── clients/                  # Client-side components
│   │   └── local_trainer.py     # Local training logic
│   ├── data/                     # Data processing
│   │   ├── datasets.py          # Dataset loading
│   │   ├── sampling.py          # Dirichlet partitioning
│   │   └── offline_dataset.py   # Offline preprocessed datasets
│   ├── models/                   # Model definitions
│   │   ├── cnn.py               # CNN models
│   │   ├── mlp.py               # MLP model
│   │   ├── vit.py               # Vision Transformer
│   │   └── lora.py              # LoRA implementation
│   └── utils/                    # Utilities
│       ├── paths.py             # Path management
│       ├── checkpoint.py        # Checkpoint management
│       ├── communication.py     # Communication statistics
│       ├── evaluation.py        # Evaluation functions
│       └── logger.py            # Logging utilities
│
├── data/                         # Datasets
│   ├── cifar/                   # CIFAR-10
│   ├── cifar100/                # CIFAR-100
│   ├── mnist/                   # MNIST
│   └── preprocessed/            # Offline preprocessed data
│
├── scripts/                      # Executable scripts
│   ├── train/                   # Training scripts
│   │   ├── run_fedavg_cifar.sh
│   │   ├── run_fedlora_cifar100.sh
│   │   └── run_fedsdg_cifar100.sh
│   ├── preprocess/              # Data preprocessing
│   └── analysis/                # Analysis tools
│
├── logs/                         # TensorBoard logs
├── save/                         # Saved models and results
│   ├── checkpoints/
│   ├── models/
│   ├── objects/
│   └── summaries/
│
├── docs/                         # Documentation
│   ├── algorithms/              # Algorithm documentation
│   ├── user_guides/             # User guides
│   └── technical_reports/       # Technical reports
│
├── main.py                       # Main entry point
├── options.py                    # Argument parser
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-repo/Federated-Learning-PyTorch.git
cd Federated-Learning-PyTorch

# Create virtual environment (optional but recommended)
conda create -n fl python=3.10
conda activate fl

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

#### FedAvg (Baseline)
```bash
# Using script
bash scripts/train/run_fedavg_cifar.sh

# Or direct command
python main.py --alg fedavg --model cnn --dataset cifar --epochs 100 --gpu 0
```

#### FedLoRA (Parameter-Efficient)
```bash
# Using script
bash scripts/train/run_fedlora_cifar100.sh

# Or direct command
python main.py \
    --alg fedlora \
    --model vit \
    --model_variant pretrained \
    --dataset cifar100 \
    --use_offline_data \
    --offline_data_root ./data/preprocessed/ \
    --epochs 100 \
    --lora_r 8 \
    --lora_alpha 16 \
    --gpu 0
```

#### FedSDG (Dual-Path Personalization)
```bash
# Using script
bash scripts/train/run_fedsdg_cifar100.sh

# Or direct command
python main.py \
    --alg fedsdg \
    --model vit \
    --model_variant pretrained \
    --dataset cifar100 \
    --use_offline_data \
    --epochs 100 \
    --lora_r 8 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --server_agg_method fedavg \
    --gpu 0
```

### 3. Monitor Training

```bash
tensorboard --logdir=./logs
```

## 📊 Supported Algorithms

| Algorithm | Description | Communication Efficiency |
|-----------|-------------|-------------------------|
| FedAvg | Standard federated averaging | Baseline (100%) |
| FedLoRA | Low-rank adaptation | ~3.5% of full model |
| FedSDG | Dual-path with gating | ~3.5% of full model |

## 🔧 Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--alg` | Algorithm: fedavg, fedlora, fedsdg | fedavg |
| `--model` | Model: mlp, cnn, vit | mlp |
| `--model_variant` | scratch or pretrained (ViT only) | scratch |
| `--dataset` | Dataset: mnist, cifar, cifar100 | mnist |
| `--epochs` | Number of communication rounds | 10 |
| `--num_users` | Number of clients | 100 |
| `--frac` | Client participation fraction | 0.1 |
| `--local_ep` | Local training epochs | 10 |
| `--local_bs` | Local batch size | 10 |
| `--lr` | Learning rate | 0.01 |
| `--dirichlet_alpha` | Non-IID parameter (smaller=more heterogeneous) | 0.5 |
| `--lora_r` | LoRA rank (FedLoRA/FedSDG) | 8 |
| `--lora_alpha` | LoRA scaling factor | 16 |
| `--use_offline_data` | Use preprocessed data | False |
| `--gpu` | GPU ID (-1 for CPU) | -1 |

## 📁 Data Preparation

### Online Mode (Default)
Datasets are automatically downloaded when first used.

### Offline Mode (Recommended for large datasets)
```bash
# Preprocess CIFAR-100 to 224x224
python src/preprocess_cifar100.py --image_size 224

# Use with offline data
python main.py --use_offline_data --offline_data_root ./data/preprocessed/ ...
```

## 📈 Output Files

After training, find results in:

- **TensorBoard logs**: `./logs/<experiment_name>/`
- **Training summary**: `./save/summaries/<experiment>_summary.txt`
- **Saved models**: `./save/models/`
- **Training objects**: `./save/objects/`
- **Checkpoints**: `./save/checkpoints/`

## 📚 Documentation

See the `docs/` directory for detailed documentation:

- [Algorithm Design](docs/algorithms/) - FedAvg, FedLoRA, FedSDG design docs
- [User Guides](docs/user_guides/) - Data preprocessing, pretrained models
- [Technical Reports](docs/technical_reports/) - Bug reports, optimization

## 🔗 Legacy Support

The original source files in `src/` are preserved for backward compatibility.
New code should use the modular structure in `fl/`.

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{federated-learning-pytorch,
  title={Federated Learning PyTorch},
  author={FL Research Team},
  year={2024},
  url={https://github.com/your-repo/Federated-Learning-PyTorch}
}
```

## 📄 License

MIT License


