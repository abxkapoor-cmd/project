from cowc_dataset import COWCPatchDataset

dataset = COWCPatchDataset("data/Toronto_ISPRS/train")

for i in range(len(dataset)):
    if dataset.files[i].startswith("car."):
        img, target = dataset[i]
        print("Found a positive patch at index", i)
        print("Filename:", dataset.files[i])
        print("Image shape:", img.shape)
        print("Target:", target)
        break