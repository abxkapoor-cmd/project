import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class COWCPatchDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: the folder containing the .png patches
        e.g. "data/cowc/Toronto_ISPRS/train"
        """
        self.root_dir = root_dir
        # Get every png file in the folder
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".png")]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.root_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__(idx+1)
        img_tensor = self.to_tensor(img)

        width, height = img.size
        is_positive = filename.startswith("car.")

        if is_positive:
            # Place a box roughly centered on the patch, sized like a typical car
            # (COWC's own paper says cars are about 24-48px long, 10-20px wide)
            box_w, box_h = 40, 15
            cx, cy = width // 2, height // 2
            xmin = max(0, cx - box_w // 2)
            ymin = max(0, cy - box_h // 2)
            xmax = min(width, cx + box_w // 2)
            ymax = min(height, cy + box_h // 2)

            boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)  # 1 = "car"

            # Approximate mask: a filled rectangle matching the box
            mask = torch.zeros((height, width), dtype=torch.uint8)
            mask[ymin:ymax, xmin:xmax] = 1
            masks = mask.unsqueeze(0)  # shape: [1, H, W]
        else:
            # No car in this patch — empty boxes/labels/masks
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target