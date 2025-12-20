import pandas as pd
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.text_values = list[str]()
        self.label_values = list[int]()

        for row in dataframe.itertuples(index=False):
            self.text_values.append(row.text)
            self.label_values.append(int(row.spam))

    def __len__(self) -> int:
        return len(self.label_values)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return (self.text_values[index], self.label_values[index])
