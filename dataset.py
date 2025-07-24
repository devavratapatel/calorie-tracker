from torchvision import datasets, transforms
from pathlib import Path

data_dir = Path("data")

def get_dataset(train_transforms: transforms, test_transforms:transforms):
    train_data = datasets.Food101(root=data_dir,
                                  split="train",
                                  transform=train_transforms,
                                  download=True)
    
    test_data = datasets.Food101(root=data_dir,
                                  split="test",
                                  transform=test_transforms,
                                  download=True)
    
    return train_data, test_data