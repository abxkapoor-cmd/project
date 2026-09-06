from torchgeo.datasets import COWCDetection

dataset = COWCDetection(root = "./data", split = "train", download = True)
print(f"Downloaded {len(dataset)} samples")