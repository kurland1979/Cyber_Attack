import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    # Model evaluation utilities: computes metrics, confusion matrix, and ROC curves.
    
    def __init__(self,model,test_loader):
        # Initialize with trained model and test DataLoader.
        
        self.model = model
        self.test_loader = test_loader
        
    def train_test(self):
        # Evaluate the model on the test set.
        # Calculates predictions, probabilities, confusion matrix, and ROC-AUC scores for all classes.
        # Stores results as class attributes and prints a summary.
        
        self.model.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_test_batch, y_test_batch in self.test_loader: 
                outputs = self.model(X_test_batch)
                proba_test = torch.softmax(outputs,dim=1)
                pred_test = torch.argmax(proba_test,dim=1)
                
                all_probs.append(proba_test)
                all_preds.append(pred_test)
                all_labels.append(y_test_batch)
                
        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        all_probs_np = all_probs.cpu().numpy()
        all_preds_np = all_preds.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()
        
        # Confusion matrix
        cm = confusion_matrix(all_labels_np, all_preds_np)
        print(classification_report(all_labels_np, all_preds_np))
        
        n_classes = 15
        y_test_binary = label_binarize(all_labels_np, classes=range(n_classes))
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], all_probs_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        macro_roc_auc = sum(roc_auc.values()) / n_classes
        print(f"ROC-AUC Score (macro-average): {macro_roc_auc:.4f}")
        
        self.cm = cm
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.n_classes = n_classes
        
    def plot_confusion_matrix(self):
        # Plot and save the confusion matrix heatmap.
        
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('matrix_graph.png') 
        try:
            plt.show()  
        except:
            pass  
        plt.close()
        
    def plot_roc_curves(self):
        # Plot and save ROC curves for all classes, including macro-average AUC.
        all_fpr = []
        all_tpr = []
    
        for i in range(self.n_classes):
            all_fpr.extend(self.fpr[i])
            all_tpr.extend(self.tpr[i])
    
        macro_auc = sum(self.roc_auc.values()) / self.n_classes
    
        plt.figure(figsize=(8, 6))
    
        for i in range(self.n_classes):
            plt.plot(self.fpr[i], self.tpr[i], color='purple', alpha=0.3)
    
        plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - Macro Average AUC = {macro_auc:.3f}')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_graph.png')  
        try:
            plt.show()  
        except:
            pass  
        plt.close()
        