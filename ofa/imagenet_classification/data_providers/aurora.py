
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from base_provider import DataProvider

__all__ = ["ImagenetDataProvider"]


class AuroraDataProvider(DataProvider):
    DEFAULT_PATH = "/dataset/imagenet"

    def __init__(
        self,
        save_path=None,
        train_batch_size=256,
        test_batch_size=256,
        valid_size=None,
        n_worker=32,
        resize_scale=0.08,
        distort_color=None,
        image_size=224,
        num_replicas=None,
        rank=None,
        seed=42
    ):
        self._save_path = save_path

        self.image_size = image_size  # int or list of int    
        self.distort_color = "None" if distort_color is None else distort_color
        self.resize_scale = resize_scale

        self.dataset_mean = [0.23280394, 0.24616548, 0.26092353]
        self.dataset_std = [0.16994016, 0.17286949, 0.16250615]

        self._valid_transform_dict = {}

        train_dataset = AuroraDataset(
            root=data_dir,
            transform=train_transforms,
            download=True,
            split="train",
            test_size=0.2,  # 20% of the data is used for testing
            random_seed=seed
        )
        test_dataset = AuroraDataset(
            root=data_dir,
            transform=test_transforms,
            split="test",
            test_size=0.2,
            random_seed=seed
        )

    @staticmethod
    def name():
        return "aurora"

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W
    @property
    def n_classes(self):
        return 2

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/dataset/imagenet")
        return self._save_path

    @property
    def data_url(self):
        raise ValueError("unable to download %s" % self.name())

    def train_dataset(self, _transforms):
        return datasets.ImageFolder(self.train_path, _transforms)

    def test_dataset(self, _transforms):
        return datasets.ImageFolder(self.valid_path, _transforms)

    @property
    def train_path(self):
        return os.path.join(self.save_path, "train")

    @property
    def valid_path(self):
        return os.path.join(self.save_path, "val")


    def build_train_transform(self, image_size=None, print_log=True):
        return transforms.Compose([
                transforms.PILToTensor(),
                transforms.ToDtype(torch.uint8, scale=True),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(self.dataset_mean, self.dataset_std)
            ])

    def build_valid_transform(self, image_size=None):
        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize(image_size),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(self.dataset_mean, self.dataset_std)
        ])



from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets
from typing import Union, Callable, Optional, Tuple, List
from pathlib import Path
from PIL import Image
import os

class AuroraDataset(datasets.VisionDataset):
    """
    Aurora Dataset with a single folder structure.
    
    Args:
        root (str or pathlib.Path): Root directory of the dataset containing `aurora` and `no_aurora` folders.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        split (str, optional): One of ['train', 'val', 'test'].
        test_size (float, optional): Fraction of the dataset used for testing (default 0.2).
        val_size (float, optional): Fraction of the training set used for validation (default 0.1).
        random_seed (int, optional): Seed for reproducibility.
    """

    classes = ["no_aurora", "aurora"]

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_size: Optional[float] = 0.2,
        val_size: Optional[float] = 0.1,
        random_seed: Optional[int] = 101
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.data, self.targets = self._load_data()

        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of ['train', 'val', 'test']")

        # Split into train and test
        train_data, test_data, train_targets, test_targets = train_test_split(
            self.data, self.targets, test_size=test_size, stratify=self.targets, random_state=random_seed
        )

        # Split train further into train and validation
        train_data, val_data, train_targets, val_targets = train_test_split(
            train_data, train_targets, test_size=val_size, stratify=train_targets, random_state=random_seed
        )

        # Assign data based on requested split
        if split == "train":
            self.data, self.targets = train_data, train_targets
        elif split == "val":
            self.data, self.targets = val_data, val_targets
        else:
            self.data, self.targets = test_data, test_targets

    def _download_data(self) -> None:
        """
        Downloads the dataset from Kaggle and extracts it to the root directory.

        Args:
            kaggle_dataset (str): The Kaggle dataset identifier (e.g., "username/dataset-name").
        """

        # Set up Kaggle API and download the dataset
        kaggle_dataset = "villehokkinen/aurora"
        try:
            if not os.path.exists(str(self.root)):
                kaggle.api.dataset_download_files(
                    kaggle_dataset,
                    path=str(self.root),
                    unzip=True,
                )
                print(f"Dataset {kaggle_dataset} downloaded and extracted to {self.root}.")
            else:
                print(f"Dataset {kaggle_dataset} already exists in {self.root}")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset {kaggle_dataset} from Kaggle: {e}")


    def _load_data(self) -> Tuple[List[str], List[int]]:
        """
        Loads the dataset from the specified root directory.
        Expects subdirectories `aurora` and `no_aurora`.
        """
        data = []
        targets = []

        for class_index, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root, class_name)
            if not os.path.exists(class_path):
                raise RuntimeError(f"Class folder {class_name} not found at {class_path}")

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(("png", "jpg", "jpeg", "bmp", "tiff")):
                    data.append(os.path.join(class_path, file_name))
                    targets.append(class_index)

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Returns an image and its label.
        """
        img_path = self.data[index]
        target = self.targets[index]

        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

train_data = AuroraDataset(
    root="/home/vilatsut/Desktop/archive", 
    split="train"
)
val_data = AuroraDataset(
    root="/home/vilatsut/Desktop/archive", 
    split="val"
)
train_data = AuroraDataset(
    root="/home/vilatsut/Desktop/archive", 
    split="train"
)

