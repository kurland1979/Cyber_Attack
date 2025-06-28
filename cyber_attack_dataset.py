from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import config

class CyberAttackDataset(Dataset):
    # Custom PyTorch dataset for cyber attack data.
    
    def __init__(self, X, y):
        # Initialize dataset with features and labels.
        self.X = X  
        self.y = y 

    def __len__(self):
        # Return the number of samples in the dataset.
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve a single sample and its label by index.
        
        return self.X[idx], self.y[idx]
    
def prepare_datasets(df, selected_features, test_size=0.15, val_size=0.15, batch_size=config.BATCH_SIZE):
    # Split data into train/validation/test sets, normalize, and create DataLoaders.
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    selected_features : list of str
        Features to use for training.
    test_size : float, optional
        Proportion of data for test set.
    val_size : float, optional
        Proportion of data for validation set.
    batch_size : int, optional
        Batch size for DataLoader.

    Returns
    -------
    tuple
        train_loader, val_loader, test_loader, scaler
    """
    X = df[selected_features]
    y = df['target_encoded']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    # normalization
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Conversion to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    # Dataset preparation
    train_dataset = CyberAttackDataset(X_train_tensor, y_train_tensor)
    val_dataset = CyberAttackDataset(X_val_tensor, y_val_tensor)
    test_dataset = CyberAttackDataset(X_test_tensor, y_test_tensor)
    
    # DataLoader preparation
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler
        

       