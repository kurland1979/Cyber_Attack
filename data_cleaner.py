import pandas as pd 
import numpy as np

class DataCleaner():
    # DataCleaner class for loading, merging, and cleaning multiple CSV files related to cyber attack datasets.
    # Handles file loading, column cleaning, target encoding, and duplicate removal.
    
    def __init__(self):
        # Initialize DataCleaner instance with an empty DataFrame.
        self.df = None
        
    def get_files(self):
        # Return a list of source CSV file paths for loading the raw data.
        base_path = r'C:\Users\User\OneDrive\מסמכים\Cyber_Attack\archive (20).csv'
        files = [
            base_path + r'\Wednesday-workingHours.pcap_ISCX.csv',
            base_path + r'\Tuesday-WorkingHours.pcap_ISCX.csv',
            base_path + r'\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            base_path + r'\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            base_path + r'\Monday-WorkingHours.pcap_ISCX.csv',
            base_path + r'\Friday-WorkingHours-Morning.pcap_ISCX.csv',
            base_path + r'\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            base_path + r'\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
        ]
        return files
        
    def load_and_concat(self):
        # Load all CSV files from the source list and concatenate them into a single DataFrame.
        files = self.get_files()  
        dataframes = []
        for file in files:
            df = pd.read_csv(file)
            dataframes.append(df)
        self.df = pd.concat(dataframes, ignore_index=True)
        
    def drop_columns(self):
        # Drop columns that are irrelevant for modeling. Handle missing values (NaN) and infinite values (inf).
        columns = [
            ' Destination Port',' Fwd Packet Length Max',' Fwd Packet Length Min',
            'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Flow IAT Max',' Flow IAT Min',
            ' Fwd IAT Max',' Fwd IAT Min', ' Bwd IAT Max', ' Bwd IAT Min',
            ' Min Packet Length', ' Max Packet Length',' Active Max',' Active Min',
            ' Idle Max', ' Idle Min'
        ]
        
        self.df.drop(columns=columns,inplace=True)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        
    def fix_column_names(self):
        # Clean column names: strip spaces, lowercase, replace spaces and slashes with underscores.
        # Encode the target (label) column as numeric, and remove the original label column.
        
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/','_')
        self.df['target_encoded'] = self.df['label'].astype('category').cat.codes
        self.df.drop(columns='label',inplace=True)
        
    def removing_duplicates(self):
        self.df = self.df.drop_duplicates()

        
    


