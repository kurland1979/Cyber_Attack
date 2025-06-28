# Cyber Attack Detection Using Deep Learning

A neural network approach to detect 15 different types of cyber attacks in network traffic.

## 🎯 Project Overview

This project implements a deep neural network to identify cyber attacks with high accuracy, specifically addressing the challenge of detecting rare attack types in highly imbalanced datasets.

### Key Results:
- **91% Overall Accuracy**
- **94-100% Recall** for rare attack types
- **0.9958 ROC-AUC Score**

## 📁 Project Structure

```
cyber-attack-detection/
├── data_cleaner.py         # Data preprocessing
├── config.py              # Configuration parameters
├── feature_engineering.py  # Feature processing
├── neural_net_builder.py   # Model architecture
├── cyber_attack_dataset.py # Dataset handling
├── trainer.py              # Training logic
├── evaluator.py            # Model evaluation
├── main.py                 # Main training script
├── Cyber_Attack_Detection_Report.pdf  # Detailed project report
└── results/
    ├── graphs/            # Training visualizations
    └── cyber_attack_model_final.pt  # Trained model
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

### Running the Project
```bash
python main.py
```

## 🛠 Configuration

Edit `config.py` to modify:
- `EPOCHS`: Number of training epochs (default: 100)
- `BATCH_SIZE`: Batch size (default: 512)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `HIDDEN_LAYERS`: Network architecture

## 📊 Model Architecture

- **Input**: 50 features
- **Hidden Layers**: [256, 224, 192, 160, 128, 96, 64]
- **Output**: 15 classes
- **Regularization**: Dropout (0.3) + Batch Normalization
- **Optimizer**: Adam with Learning Rate Scheduler

## 📈 Results

The model successfully addresses class imbalance:
- Improved minority class recall from 5% → 100%
- Maintains high precision for majority classes
- Practical for real-world deployment

## 📝 Documentation

For detailed analysis and methodology, see: [Project Report](Cyber_Attack_Detection_Report.pdf)

## 👩‍💻 Author

Marina Kurland

## 🙏 Acknowledgments

First professional deep learning project, completed with GPU support from Lightning AI.

## Dataset Access
Dataset download: https://drive.google.com/drive/folders/1i1dAEFAEjNcT9PTER_8qqk6vklWj_iJV?usp=sharing

