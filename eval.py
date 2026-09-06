import random
random.seed(42)
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from cowc_dataset import COWCPatchDataset


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def move_target_to_device(target, device):
    new_target = {}
    for key, value in target.items():
        new_target[key] = value.to(device)
    return new_target


# Rebuild the same eval split used during training
dataset = COWCPatchDataset("data/Toronto_ISPRS/train")
indexes = list(range(len(dataset)))
random.shuffle(indexes)  
eval_dataset = Subset(dataset, indexes[2000:2500])

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)

# Model setup
model = maskrcnn_resnet50_fpn(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

checkpoint = torch.load("checkpoint_epoch_9.pth")
model.load_state_dict(checkpoint)
model.eval()

metric = MeanAveragePrecision()

with torch.no_grad():
    for images, targets in eval_dataloader:
        images = [img.to(device) for img in images]
        targets = [move_target_to_device(t, device) for t in targets]

        predictions = model(images)

        metric.update(predictions, targets)

results = metric.compute()
print(results)