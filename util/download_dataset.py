import fiftyone as fo # Ignore the import error as it will be run in virtual env, where the dependency will be present

dataset = fo.zoo.load_zoo_dataset(
    "open-images-v7",
    split="train",
    abel_types=["detections"],
    classes=["Vehicle registration plate"],
    max_samples=10,
)
