from cowc_dataset import COWCPatchDataset
from torch.utils.data import DataLoader
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import time
from torch.utils.data import Subset
import random
random.seed(42)


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

dataset = COWCPatchDataset("data/Toronto_ISPRS/train")
indexes = list(range(len(dataset)))
random.shuffle(indexes)
small_dataset = Subset(dataset, indexes[0:2000])
eval_dataset = Subset(dataset, indexes[2000:2500])

dataloader = DataLoader(
    small_dataset,
    batch_size = 2,
    shuffle = True,
    collate_fn = collate_fn
)

def move_target_to_device(target,device):
    new_target = {}
    for key,value in target.items():
        new_target[key] = value.to(device)
    return new_target


model = maskrcnn_resnet50_fpn(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
model.to(device)
print("Model is on: ", next(model.parameters()).device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

model.train()
num_epochs = 10
for epoch in range(num_epochs):
    start = time.time()
    for i, (images, targets) in enumerate(dataloader):
        # if i == 100:
        #    end = time.time()
        #    print ("Elapsed time: ", end-start)
        #    break 
        images = [img.to(device) for img in images]
        targets =[move_target_to_device(target, device) for target in targets]

        losses = model(images,targets)
        total_loss = 0
        for loss in losses.values():
            total_loss += loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print("Loss: ", total_loss.item())
    
    print(f"Epoch {epoch} finished in {time.time() - start:.1f} seconds")
    torch.save(model.state_dict(),f"checkpoint_epoch_{epoch}.pth")




