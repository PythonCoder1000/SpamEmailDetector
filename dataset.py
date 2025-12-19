import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.X = list[str]()
        self.y = list[int]()

        for row in df.itertuples(index=False):
            self.X.append(row.text)
            self.y.append(row.spam)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx) -> tuple[str, int]:
        return (self.X[idx], self.y[idx])
