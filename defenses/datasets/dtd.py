import os
import os.path as osp
import pathlib
from typing import Any, Callable, Optional, Tuple

import PIL.Image

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import defenses.config as cfg

class DTD(VisionDataset):
    """`Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        partition (int, optional): The dataset partition. Should be ``1 <= partition <= 10``. Defaults to ``1``.

            .. note::

                The partition only changes which split each image belongs to. Thus, regardless of the selected
                partition, combining all splits will result in all images.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    _MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        root = osp.join(cfg.DATASET_ROOT, 'DTD')
        self._split = 'train' if train else 'test'
        #self._split = verify_str_arg(split, "split", ("train", "val", "test"))

        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / type(self).__name__.lower()
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._image_files = []
        classes = []
        
        with open(self._meta_folder / f"{self._split}{1}.txt") as file:
            for line in file:
                cls, name = line.strip().split("/")
                self._image_files.append(self._images_folder.joinpath(cls, name))
                classes.append(cls)

        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]

        self.samples = self._image_files
        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split},partition=1"

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=str(self._base_folder), md5=self._MD5)