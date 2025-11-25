import os
import torch
import shutil
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use an interactive backend for real-time visuals
import matplotlib.pyplot as plt
plt.ion()
from datetime import datetime
from PIL import Image
import math
import time
import torch.optim as optim
import torch.nn.functional as F  # ✅ FIX: Import F for interpolate()

cmap = plt.cm.viridis

def parse_command():
    data_names = ['nyudepthv2', 'kitti']
    import argparse
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('--data', metavar='DATA', default='nyudepthv2', choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    parser.add_argument('--batch_size', type=int, default=8, help='Mini-batch size (default: 8)')

    if parser.parse_known_args()[0].data == 'kitti':
        from dataloaders.kitti import KITTIDataset
        modality_names = ['rgb']  # KITTI dataset only supports RGB modality
    else:
        from dataloaders.dataloader import MyDataloader
        modality_names = MyDataloader.modality_names

    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH',)
    parser.add_argument('--gpu', default='0', type=str, metavar='N', help="gpu id")
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    return args

def verify_data_loading(train_loader, val_loader):
    print(f"✅ Dataset is loading correctly!")
    print(f"✅ Training samples loaded: {len(train_loader.dataset)}")
    print(f"✅ Validation samples loaded: {len(val_loader.dataset)}")

def monitor_training_progress(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=5):
    print("🚀 Starting Training...")
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # ✅ Reset gradients before forward pass
            outputs = model(inputs)

            # Resize both outputs and targets to ensure they match
            outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
            targets = F.interpolate(targets, size=(224, 224), mode='bilinear', align_corners=False)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()  # ✅ Update model weights

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                check_gpu_usage()

        # ✅ Step learning rate scheduler
        scheduler.step()

        epoch_end = time.time()
        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} completed in {epoch_end - epoch_start:.2f} seconds. Average Loss: {avg_loss:.4f}")

        # 💾 SAVE MODEL CHECKPOINT AFTER EACH EPOCH
        save_checkpoint(model, optimizer, epoch, avg_loss)

def save_checkpoint(model, optimizer, epoch, loss):
    os.makedirs("checkpoints", exist_ok=True)  # ✅ Ensure the directory exists
    checkpoint_path = os.path.join("checkpoints", f"checkpoint_epoch_{epoch+1}.pth.tar")
    
    print(f"💾 Saving checkpoint for Epoch {epoch+1}...")
    checkpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved at: {checkpoint_path}")

def check_gpu_usage():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"⚡ GPU {i} Memory Usage: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        print("⚠️ GPU not available, running on CPU.")


if __name__ == "__main__":
    args = parse_command()

    # 🔹 Step 1: Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Running FastDepth with dataset: {args.data} and batch size: {args.batch_size}")

    # 🔹 Step 2: Load dataset (KITTI or NYU Depth V2)
    if args.data == "kitti":
        from dataloaders.kitti import KITTIDataset
        kitti_dataset_path = "E:/caterpillar/Caterpillar Project/KITTI_Dataset"

        train_loader = torch.utils.data.DataLoader(
            KITTIDataset(root_dir=kitti_dataset_path, split='train'),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            KITTIDataset(root_dir=kitti_dataset_path, split='val'),
            batch_size=args.batch_size, shuffle=False
        )
    else:
        from dataloaders.dataloader import MyDataloader
        train_loader = torch.utils.data.DataLoader(
            MyDataloader(split='train'),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            MyDataloader(split='val'),
            batch_size=args.batch_size, shuffle=False
        )

    verify_data_loading(train_loader, val_loader)

    # 🔹 Step 3: Initialize Model
    from models.fastdepth import FastDepth
    model = FastDepth().to(device)

    # 🔹 Step 4: Define Loss Function & Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 🔹 Step 5: Start Training
    monitor_training_progress(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=5)

           #Evaluate the Model on the Validation Set
# def evaluate_model(model, val_loader, criterion, device):
#     print("📊 Evaluating Model on Validation Set...")
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(val_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
#             targets = F.interpolate(targets, size=(224, 224), mode='bilinear', align_corners=False)
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()

#     avg_loss = total_loss / len(val_loader)
#     print(f"✅ Validation Loss: {avg_loss:.4f}")

# evaluate_model(model, val_loader, criterion, device)


#visaul saving
import os

VISUALS_PATH = r"E:\caterpillar\Caterpillar Project\FastDepth\fast-depth\visuals_saving"

# 🔹 Function to Save Visuals
def save_visual(fig, visuals_saving):
    os.makedirs(VISUALS_PATH, exist_ok=True)
    visual_file_path = os.path.join(VISUALS_PATH, f"{visuals_saving}.png")
    fig.savefig(visual_file_path)
    plt.close(fig)  # ✅ Prevents `plt.show()` from blocking the saving
    print(f"✅ Visual saved successfully at: {visual_file_path}")


# #visulaize the model

def visualize_predictions(model, val_loader, device, num_images=5):
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
        
    plt.show()

visualize_predictions(model, val_loader, device)

#model eval

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


#graphical visuals

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def visualize_predictions(model, val_loader, device, num_images=5):
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
        axes[i, 0].axis('off')

        axes[i, 1].imshow(depth_maps[i].cpu().squeeze(), cmap='viridis')
        axes[i, 1].set_title("Ground Truth Depth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(predicted_depths[i].cpu().squeeze(), cmap='viridis')
        axes[i, 2].set_title("Predicted Depth")
        axes[i, 2].axis('off')

    # 🔹 Save Visual and Show
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_visual(fig, f"visual_{timestamp}")

    plt.show(block=False)  # ✅ Ensures visuals are displayed without blocking
    print("✅ Visuals displayed successfully!")

# Call the function after training
visualize_predictions(model, val_loader, device)

import matplotlib.pyplot as plt

def plot_training_progress(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 🔹 Save Progress Graph
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.join(VISUALS_PATH, f'training_progress_{timestamp}.png'))
    plt.close()  # ✅ Prevent blocking during visualization
    print(f"✅ Training progress graph saved successfully at: {VISUALS_PATH}")


#savingg

torch.save(model.state_dict(), "fastdepth_trained.pth")
print("✅ Model saved successfully!")
