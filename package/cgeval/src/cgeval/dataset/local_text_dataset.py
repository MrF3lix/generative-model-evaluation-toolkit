from torch.utils.data import DataLoader
from datasets import load_dataset

from cgeval import Dataset

class LocalTextDataset(Dataset):
    def __init__(self, cfg, column_mapping = None):
        super().__init__()
        self.cfg = cfg
        self.column_mapping = column_mapping

    def load(self) -> DataLoader:
        ds = load_dataset("json", data_files=self.cfg.dataset.name)
        if not self.column_mapping == None:
            ds = ds.rename_columns(self.column_mapping)

        subset = ds['train']
        if 'dataset.samples' in self.cfg:
            subset = ds['train'].select(range(0, self.cfg.dataset.samples))

        dataloader = DataLoader(subset, batch_size=self.cfg.dataset.batch_size, shuffle=False)

        return dataloader