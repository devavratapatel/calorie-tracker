import torch
import torchvision
import torchinfo
from torchinfo import summary
from torchvision import transforms, datasets
from create_model import create_effnetb2_model
from dataset import get_dataset
from pathlib import Path

def food101_model_and_transforms(num_classes:int):
    effnetb2_food101, effnetb2_transforms = create_effnetb2_model(num_classes=num_classes)
    summary(effnetb2_food101,
        input_size=(1,3,224,224),
        col_names=["input_size","output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"])
    
    effnetb2_transforms = torchvision.transforms.Compose([
        transforms.ToTensor(),
        effnetb2_transforms,
    ])
    food101_train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.TrivialAugmentWide(),
        effnetb2_transforms,
    ])

    train_data, test_data = get_dataset(food101_train_transforms,effnetb2_transforms)

    return effnetb2_food101, train_data, test_data

def get_model_and_test_transforms(num_classes:int):
    effnetb2_food101, effnetb2_transforms = create_effnetb2_model(num_classes=num_classes)
    effnetb2_transforms = torchvision.transforms.Compose([
        transforms.ToTensor(),
        effnetb2_transforms,
    ])

    return effnetb2_food101, effnetb2_transforms

def split_dataset(dataset: datasets, split_size:float):
    length_1 = int(len(dataset) * split_size)
    length_2 = len(dataset) - length_1
    print(f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {length_1} ({int(split_size*100)}%), {length_2} ({int((1-split_size)*100)}%)")

    random_split_1, random_split_2 = torch.utils.data.random_split(dataset,
                                                   lengths=[length_1,length_2],
                                                   generator=torch.manual_seed(42))
    return random_split_1, random_split_2

def get_dataloaders(train_dir: datasets, test_dir: datasets, batch_size: int, num_workers:int):
    train_dataloader = torch.utils.data.DataLoader(train_dir,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers)
    
    test_dataloader = torch.utils.data.DataLoader(test_dir,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)
    
    return train_dataloader, test_dataloader

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswidth(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)