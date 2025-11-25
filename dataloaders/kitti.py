import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class KITTIDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.rgb_dir = os.path.join(root_dir, split, 'rgb')
        self.depth_dir = os.path.join(root_dir, split, 'depth')

        # ✅ Check if dataset directories exist
        if not os.path.exists(self.rgb_dir) or not os.path.exists(self.depth_dir):
            raise FileNotFoundError(f"❌ Dataset directories not found: {self.rgb_dir} or {self.depth_dir}")

        self.rgb_images = sorted(os.listdir(self.rgb_dir))
        self.depth_images = sorted(os.listdir(self.depth_dir))

        # ✅ Ensure the number of RGB and depth images match
        if len(self.rgb_images) != len(self.depth_images):
            print(f"⚠️ Warning: Number of RGB images ({len(self.rgb_images)}) "
                  f"and depth images ({len(self.depth_images)}) do not match!")

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_images[idx])

        # ✅ Handle missing files
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            raise FileNotFoundError(f"❌ Missing image files: {rgb_path} or {depth_path}")

        # Load and process RGB image
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))

        # Load and process Depth image
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (224, 224)).astype(np.float32)

        # ✅ Normalize depth values
        depth = depth / 255.0  # Change this if depth values are in a different range

        if self.transform:
            rgb = self.transform(rgb)
        else:
            rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize RGB

        depth = torch.tensor(depth, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return rgb, depth
