from functools import partial

import flash
# from flash.core.utilities.imports import example_requires
# from flash.image import InstanceSegmentation, InstanceSegmentationData

import torch

import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier
from torchvision import models, transforms

# example_requires("image")

# import icedata  # noqa: E402

# 1. Create the DataModule
# data_dir = icedata.pets.load_data()
data_dir = "/home/ubuntu/Sketch-Recommendation/data/kaggle_2017_ImageNet_data/ILSVRC/Data/DET/train/ILSVRC2013_train"

# datamodule = InstanceSegmentationData.from_icedata(
#     train_folder=data_dir,
#     val_split=0.1,
#     parser=partial(icedata.pets.parser, mask=True),
#     batch_size=4,
# )
datamodule = ImageClassificationData.from_folders(
    train_folder=data_dir,
    val_split=0.1,
    batch_size=32,
    transform_kwargs={"image_size": (196, 196), "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
)

print(len(datamodule.labels), datamodule.labels)
model = ImageClassifier(
    backbone="tf_efficientnetv2_l_in21ft1k", 
    # num_classes=dm.num_classes, 
    pretrained=True,
    # labels=['n01440764', 'n01514668', 'n01558993'],
    labels=datamodule.labels
    )

trainer = flash.Trainer(max_epochs=3, gpus=1, profiler="advanced")
trainer.finetune(model, datamodule=datamodule, strategy='freeze')

# 4. Detect objects in a few images!
# datamodule = InstanceSegmentationData.from_files(
#     predict_files=[
#         str(data_dir / "images/yorkshire_terrier_9.jpg"),
#         str(data_dir / "images/yorkshire_terrier_12.jpg"),
#         str(data_dir / "images/yorkshire_terrier_13.jpg"),
#     ],
#     batch_size=4,
# )


predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("instance_segmentation_model.pt")