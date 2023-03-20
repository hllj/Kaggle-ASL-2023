from torch.utils.data import Dataset

class ASLData(Dataset):
    def __init__(self, datax, datay):
        self.datax = datax
        self.datay = datay
        
    def __getitem__(self, index):
        return self.datax[index,:], self.datay[index]
        
    def __len__(self):
        return len(self.datay)