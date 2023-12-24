import fiftyone as fo

dataset = fo.zoo.load_zoo_dataset(
    "open-images-v7",
    split="train",
    abel_types=["detections"],
    classes=["Vehicle registration plate"],
    max_samples=10,
)
