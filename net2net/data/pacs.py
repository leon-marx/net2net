import os, pickle
import albumentations
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from net2net.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import net2net.data.utils as ndu

class PACSBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

class PACSGeneralBase(Dataset):
    def __init__(self, config=None, domain=None, content=None):
        self.domain = domain
        self.content = content
        self.config = config or dict()
        self._prepare()
        self._load()

    def _prepare(self):
        self.root = f"C:/Users/gooog/Desktop/Bachelor/Code/bachelor/data/PACS/{self.domain}/{self.content}/"
        namelist = os.listdir(self.root)
        with open(f"C:/Users/gooog/Desktop/Bachelor/Code/bachelor/data/meta/{self.domain}_{self.content}.txt", "w") as f:
            for name in namelist:
                if name != ".ready":
                    f.write(name + "\n")
        self._data_path = f"C:/Users/gooog/Desktop/Bachelor/Code/bachelor/data/meta/{self.domain}_{self.content}.txt"
        if not ndu.is_prepared(self.root):
            print(f"preparing {self.domain} {self.content} dataset...")
            # prep
            root = Path(self.root)
            ndu.mark_prepared(self.root)

    def _get_split(self):
        split = (
            "test" if self.config.get("test_mode", False) else "train"
        )  # default split
        return split

    def _load(self):
        self._length = len(os.listdir(self.root)) - 1

    def _load_example(self, i):
        example = dict()
        for k in self.labels:
            example[k] = self.labels[k][i]
        example["image"] = Image.open(os.path.join(self.root, example["fname"]))
        if not example["image"].mode == "RGB":
            example["image"] = example["image"].convert("RGB")
        example["image"] = np.array(example["image"])
        return example

    def _preprocess_example(self, example):
        example["image"] = example["image"] / 127.5 - 1.0
        example["image"] = example["image"].astype(np.float32)

    def __getitem__(self, i):
        example = self._load_example(i)
        self._preprocess_example(example)
        return example

    def __len__(self):
        return self._length

class PACSGeneral(PACSGeneralBase):
    """General PACS dataset with support for resizing and fixed cropping"""
    def __init__(self, config, domain, content):
        super().__init__(config, domain, content)
        self.size = config["spatial_size"]
        self.cropper = albumentations.CenterCrop(height=160,width=160)
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.preprocessor = albumentations.Compose([self.cropper, self.rescaler])

        if "cropsize" in config and config["cropsize"] < self.size:
            self.cropsize = config["cropsize"]
            self.preprocessor = albumentations.Compose([
                self.preprocessor,
                albumentations.RandomCrop(height=self.cropsize, width=self.cropsize)])

    def _preprocess_example(self, example):
        example["image"] = self.preprocessor(image=example["image"])["image"]
        example["image"] = (example["image"] + np.random.random()) / 256.  # dequantization
        example["image"] = (255 * example["image"])
        return super()._preprocess_example(example)

    def __getitem__(self, i):
        example = super().__getitem__(i)
        example['index'] = i
        return example

class _PACSGeneralTrain(PACSGeneral):
    def _get_split(self):
        return "train"

class _PACSGeneralTest(PACSGeneral):
    def _get_split(self):
        return "test"

class PACSGeneralTrain(PACSBase):
    def __init__(self, size, keys=None, domain=None, content=None):
        super().__init__()
        cfg = {"spatial_size": size}
        self.data = _PACSGeneralTrain(cfg, domain=domain, content=content)

class PACSGeneralValidation(PACSBase):
    def __init__(self, size, keys=None, domain=None, content=None):
        super().__init__()
        cfg = {"spatial_size": size}
        self.data = _PACSGeneralTest(cfg, domain=domain, content=content)

class PACSTrain(Dataset):
    """
    domain: photo, art_painting, cartoon, sketch
    content: dog, elephant, giraffe, guitar, horse, house, person
    """
    def __init__(self, size=227):
        d1 = PACSGeneralTrain(size=size, keys=["image"], domain="photo", content="dog")
        d2 = PACSGeneralTrain(size=size, keys=["image"], domain="art_painting", content="dog")
        d3 = PACSGeneralTrain(size=size, keys=["image"], domain="cartoon", content="dog")
        d4 = PACSGeneralTrain(size=size, keys=["image"], domain="sketch", content="dog")
        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example

class PACSValidation(Dataset):
    """
    domain: photo, art_painting, cartoon, sketch
    content: dog, elephant, giraffe, guitar, horse, house, person
    """
    def __init__(self, size=227):
        d1 = PACSGeneralValidation(size=size, keys=["image"], domain="photo", content="dog")
        d2 = PACSGeneralValidation(size=size, keys=["image"], domain="art_painting", content="dog")
        d3 = PACSGeneralValidation(size=size, keys=["image"], domain="cartoon", content="dog")
        d4 = PACSGeneralValidation(size=size, keys=["image"], domain="sketch", content="dog")
        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example

if __name__ == "__main__":

    d = PACSTrain(size=227)
    print("size PACSTrain:", len(d))
    d = PACSValidation(size=227)
    print("size PACSValidation:", len(d))
    x = d[0]["image"]
    print(x.shape)
    print(type(x))
    print(x.max(), x.min())
