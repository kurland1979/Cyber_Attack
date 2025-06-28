import torch
import torch.nn as nn
import torch.optim as optim
import config
import numpy as np

class Trainer:
    # Trainer for supervised model training with weighted loss and learning rate scheduling.
    
    def __init__(self, model, train_loader, val_loader):
        # Initialize the trainer with model, DataLoaders, optimizer, and scheduler.
        # Computes class weights for imbalanced data.
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader 
        
        train_labels = []
        for _, labels in self.train_loader:
            train_labels.extend(labels.numpy())
    
        unique, counts = np.unique(train_labels, return_counts=True)
        weights = len(train_labels) / (len(unique) * counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float)
        
        self.loss_fc = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, 
                        mode='min',
                        patience=3,
                        factor=0.5
                        )

    def train_model(self,epochs=config.EPOCHS):
        # Initialize the trainer with model, DataLoaders, optimizer, and scheduler.
        # Computes class weights for imbalanced data.
        
        for epoch in range(epochs):
            all_loss_train = []
            for X_train_batch, y_train_batch in self.train_loader:
                outputs = self.model(X_train_batch)
                loss_train = self.loss_fc(outputs,y_train_batch)
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()
                
                all_loss_train.append(loss_train.item())
            mean_batch_train = sum(all_loss_train) / len(all_loss_train) 
            loss_val = self.train_val()
            self.scheduler.step(loss_val)
            print(f'Epoch {epoch+1}: Loss_Train: {mean_batch_train:.4f}, Loss_Val: {loss_val:.4f}')
            
        
    def train_val(self):
        # Evaluate the model on the validation set and return mean batch loss.
        
        self.model.eval()
        with torch.no_grad():
            all_loss_val = []
            for X_val_batch, y_val_batch in self.val_loader: 
                outputs = self.model(X_val_batch)
                loss_val = self.loss_fc(outputs,y_val_batch)
                
                all_loss_val.append(loss_val.item())
            mean_batch_val = sum(all_loss_val) / len(all_loss_val)
        self.model.train()
        return mean_batch_val
    
    
        
        
        
            
                    
                
                