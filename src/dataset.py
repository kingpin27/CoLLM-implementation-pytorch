from torch.utils.data import Dataset

class MTCIRDataset(Dataset):
    def __init__(self):
        pass 
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data[idx]