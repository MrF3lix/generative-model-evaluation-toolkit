import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

from cgeval import Dataset

class LocalImageDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def load(self) -> DataLoader:
        t = transforms.Compose([
            transforms.PILToTensor()
        ])

        def image_transforms(batch):
            return {
                'input': t(batch['image']),
                'class': batch['label']
            }

        dataset = load_dataset("imagefolder", data_dir=self.cfg.dataset.name)
        dataset.set_transform(image_transforms)
        dataloader = DataLoader(dataset['train'], batch_size=self.cfg.dataset.batch_size, shuffle=False)

        return dataloader