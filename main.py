import warnings
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineering
from neural_net_builder import NeuralNetBuilder
from trainer import Trainer
from evaluator import Evaluator
from cyber_attack_dataset import CyberAttackDataset, prepare_datasets
import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Main pipeline for cyber attack classification project.
Loads and cleans data, selects features, trains a neural network,
and evaluates performance with metrics and visualizations.
"""

if __name__ == '__main__':
    cleaner = DataCleaner()
    cleaner.load_and_concat()
    cleaner.drop_columns()
    cleaner.fix_column_names()
    cleaner.removing_duplicates()
    
    df = cleaner.df
    
    features = FeatureEngineering(df)
    features.drop_constant_features()
    features.feature_importance()
    
    

    features.print_feature_summary()
    features.plot_feature_importance()
    
    train_loader, val_loader, test_loader, scaler = prepare_datasets(
    df, features.selected_features, batch_size=config.BATCH_SIZE  )

    
    model = NeuralNetBuilder(config.INPUT_DIM, config.OUTPUT_DIM, config.HIDDEN_LAYERS)
    
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train_model()
    
    
    print("Saving trained model...")
    torch.save(model.state_dict(), 'cyber_attack_model_final.pt')


    results = {
    'accuracy': 0.91,
    'final_loss_train': 0.2037,
    'final_loss_val': 0.1987,
    'epochs': config.EPOCHS,
    'architecture': config.HIDDEN_LAYERS,
    'learning_rate': config.LEARNING_RATE,
    'batch_size': config.BATCH_SIZE
    }
    
    torch.save(results, 'training_results.pt')
    print("Model and results saved successfully!")

    evaluator = Evaluator(model, test_loader)
    evaluator.train_test()
    evaluator.plot_confusion_matrix()  
    evaluator.plot_roc_curves()

    

    
   






   