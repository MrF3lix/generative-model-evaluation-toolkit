from torch.utils.data import DataLoader
from datasets import load_dataset

from cgeval import Dataset

class HuggingfaceDataset(Dataset):
    def __init__(self, cfg, column_mapping = None):
        super().__init__()
        self.cfg = cfg
        self.column_mapping = column_mapping

    def load(self) -> DataLoader:
        ds = load_dataset(self.cfg.dataset.name)
        if not self.column_mapping == None:
            ds = ds.rename_columns(self.column_mapping)

        dataloader = DataLoader(ds['train'].select(range(0, self.cfg.dataset.samples)), batch_size=self.cfg.dataset.batch_size, shuffle=False)

        return dataloader