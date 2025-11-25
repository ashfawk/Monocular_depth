import os
import torch
import shutil
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from datetime import datetime
from PIL import Image
import math
import time
import torch.optim as optim
import torch.nn.functional as F

# Visuals saving path
VISUALS_PATH = r"E:\caterpillar\Caterpillar Project\FastDepth\fast-depth\visuals_saving"

# Dataset Parsing
def parse_command():
    data_names = ['nyudepthv2', 'kitti']
    import argparse
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('--data', default='nyudepthv2', choices=data_names)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', default='0', type=str)
    parser.set_defaults(cuda=True)
    return parser.parse_args()

# Visual Saving
def save_visual(fig, visuals_saving):
    os.makedirs(VISUALS_PATH, exist_ok=True)
    visual_file_path = os.path.join(VISUALS_PATH, f"{visuals_saving}.png")
    fig.savefig(visual_file_path)
    plt.close(fig)
    print(f"✅ Visual saved at: {visual_file_path}")

# Visualizing Predictions
def visualize_predictions_vs_groundtruth(model, val_loader, device, num_images=5):
    model.eval()
    images, depth_maps = next(iter(val_loader))
    images, depth_maps = images[:num_images].to(device), depth_maps[:num_images].to(device)

    with torch.no_grad():
        predicted_depths = model(images)
        predicted_depths = F.interpolate(predicted_depths, size=(224, 224), mode='bilinear', align_corners=False)

    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))
    for i in range(num_images):
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title("Input Image")
        axes[i, 1].imshow(depth_maps[i].cpu().squeeze(), cmap='viridis')
        axes[i, 1].set_title("Ground Truth Depth")
        axes[i, 2].imshow(predicted_depths[i].cpu().squeeze(), cmap='viridis')
        axes[i, 2].set_title("Predicted Depth")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_visual(fig, f"visual_{timestamp}")
    plt.show(block=False)
    print("✅ Visuals displayed successfully!")

# Training Monitoring
def monitor_training_progress(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=5):
    print("🚀 Starting Training...")
    model.to(device).train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
            targets = F.interpolate(targets, size=(224, 224), mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, avg_loss)
        scheduler.step()

# Saving Checkpoints
def save_checkpoint(model, optimizer, epoch, loss):
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", f"checkpoint_epoch_{epoch+1}.pth.tar")
    torch.save({
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss
    }, checkpoint_path)
    print(f"✅ Checkpoint saved at: {checkpoint_path}")

# Model Evaluation
def absolute_relative_error(pred, target):
    return torch.mean(torch.abs(pred - target) / target)

def evaluate_model(model, val_loader, device):
    model.eval()
    total_error = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
            targets = F.interpolate(targets, size=(224, 224), mode='bilinear', align_corners=False)

            total_error += absolute_relative_error(outputs, targets).item()

    avg_error = total_error / len(val_loader)
    print(f"✅ Absolute Relative Error: {avg_error:.4f}")

# Main Execution
if __name__ == "__main__":
    args = parse_command()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.fastdepth import FastDepth
    model = FastDepth().to(device)

    from dataloaders.dataloader import MyDataloader
    train_loader = torch.utils.data.DataLoader(MyDataloader(root='E:\caterpillar\Caterpillar Project\FastDepth\data', split='train'), batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(MyDataloader(root='E:\caterpillar\Caterpillar Project\FastDepth\data', split='val'), batch_size=args.batch_size, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    monitor_training_progress(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=5)
    visualize_predictions_vs_groundtruth(model, val_loader, device)
    evaluate_model(model, val_loader, device)

    torch.save(model.state_dict(), "fastdepth_trained.pth")
    print("✅ Model saved successfully!")
