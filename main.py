import os
import torch
import torchvision
from torchvision import transforms
from torch import nn
from utils import food101_model_and_transforms, get_dataloaders, split_dataset, save_model
from engine import train

if __name__ == "__main__":
    model, train_data, test_data = food101_model_and_transforms(num_classes=101)

    food101_class_names = train_data.classes
    print(food101_class_names)

    # Skip splitting dataset if you have a decent GPU, split dataset if you don't
    # train_data, _ = split_dataset(train_data, 0.2)
    # test_data, _ = split_dataset(test_data, 0.2)

    batch_size = 32
    num_workers = 0

    train_dataloader, test_dataloader = get_dataloaders(train_data, test_data, batch_size, num_workers)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    #train_dataloader, test_dataloader = train_dataloader.to(device), test_dataloader.to(device)
    model = model.to(device)

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device
    )

    # plot_loss_curves(results)

    model_path = "food101_20_percent_trained.pth"

    save_model(model=model, target_dir="models", model_name=model_path)

    with open("class_names.txt", "w") as f:
        print(f"[INFO] Saving Food101 class names to class_names.txt")
        f.write("\n".join(food101_class_names))
