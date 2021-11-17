import os, pickle
import albumentations
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset

from net2net.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import net2net.data.utils as ndu


def pacs_contents_one_hot(array):
    names = np.unique(array)
    swap = {
        "dog": np.array([1, 0, 0, 0, 0, 0, 0]),
        "elephant": np.array([0, 1, 0, 0, 0, 0, 0]),
        "giraffe": np.array([0, 0, 1, 0, 0, 0, 0]),
        "guitar": np.array([0, 0, 0, 1, 0, 0, 0]),
        "horse": np.array([0, 0, 0, 0, 1, 0, 0]),
        "house": np.array([0, 0, 0, 0, 0, 1, 0]),
        "person": np.array([0, 0, 0, 0, 0, 0, 1]),
    }
    new = np.zeros((len(array), 7))
    for i, a in enumerate(array):
        new[i] = swap[a]
    return new


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
    def __init__(self, config=None, domain=None, contents=None):
        if os.name == "nt":
            self.repo_path = f"C:/Users/gooog/Desktop/Bachelor/Code/bachelor/"
        else:
            self.repo_path = f"/home/tarkus/leon/bachelor/"
        self.domain = domain
        self.contents = contents
        self.config = config or dict()
        self._prepare()
        self._load()

    def _prepare(self):
        self.root = self.repo_path + f"data/PACS/{self.domain}/"
        for i, content in enumerate(self.contents):
            namelist = os.listdir(self.root + f"{content}/")
            if i == 0:
                with open(self.repo_path + f"data/meta/{self.domain}.txt", "w") as f:
                    for name in namelist:
                        if name != ".ready":
                            f.write(f"{content}/" + name + "\n")
            else:
                with open(self.repo_path + f"data/meta/{self.domain}.txt", "a") as f:
                    for name in namelist:
                        if name != ".ready":
                            f.write(f"{content}/" + name + "\n")
            if not ndu.is_prepared(self.root + f"{content}/"):
                print(f"preparing {self.domain} {content} dataset...")
                # prep
                root = Path(self.root + f"{content}/")
                ndu.mark_prepared(self.root + f"{content}/")
        self._data_path = self.repo_path + f"data/meta/{self.domain}.txt"

    def _get_split(self):
        split = (
            "test" if self.config.get("test_mode", False) else "train"
        )  # default split
        return split

    def _load(self):
        self.full_length = 0
        for content in self.contents:
            self.full_length += len(os.listdir(self.root + f"{content}/")) - 1
        self._data = np.loadtxt(self._data_path, dtype=str)
        split = self._get_split()
        if split == "train":
            self.split_indices = [0, int(0.9*self.full_length)]  # changed 0.7 to 0.9 because "val" is not supported atm
        elif split == "val":
            self.split_indices = [int(0.7*self.full_length), int(0.9*self.full_length)]
        elif split == "test":
            self.split_indices = [int(0.9*self.full_length), -1]
        self.labels = {
            "fname": self._data[self.split_indices[0] : self.split_indices[1]],
            "domain": np.array([str(self.domain)] * self._data[self.split_indices[0] : self.split_indices[1]].shape[0]),
            "content": pacs_contents_one_hot(np.vectorize(lambda s: s.split("/")[0])(self._data[self.split_indices[0] : self.split_indices[1]]))
        }
        self._length = self.labels["fname"].shape[0]
        if True:
            print("")
            print("Domain:", self.domain)
            print("split:", split)
            print("full length:", self.full_length)
            print("split indices:", self.split_indices)
            print("shape of fname:", self.labels["fname"].shape)
            print("unique domains:", np.unique(self.labels["domain"]))
            print("unique contents:", np.unique(self.labels["content"], axis=0))
            print("")

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
    def __init__(self, config, domain, contents):
        super().__init__(config, domain, contents)
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
    def __init__(self, size, keys=None, domain=None, contents=None):
        super().__init__()
        cfg = {"spatial_size": size}
        self.data = _PACSGeneralTrain(cfg, domain=domain, contents=contents)

class PACSGeneralValidation(PACSBase):
    def __init__(self, size, keys=None, domain=None, contents=None):
        super().__init__()
        cfg = {"spatial_size": size}
        self.data = _PACSGeneralTest(cfg, domain=domain, contents=contents)

class PACSTrain(Dataset):
    """
    domain: photo, art_painting, cartoon, sketch
    content: dog, elephant, giraffe, guitar, horse, house, person
    """
    def __init__(self, size=256):
        contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        d1 = PACSGeneralTrain(size=size, keys=["image"], domain="photo", contents=contents)
        d2 = PACSGeneralTrain(size=size, keys=["image"], domain="art_painting", contents=contents)
        d3 = PACSGeneralTrain(size=size, keys=["image"], domain="cartoon", contents=contents)
        d4 = PACSGeneralTrain(size=size, keys=["image"], domain="sketch", contents=contents)
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
    def __init__(self, size=256):
        contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        d1 = PACSGeneralValidation(size=size, keys=["image"], domain="photo", contents=contents)
        d2 = PACSGeneralValidation(size=size, keys=["image"], domain="art_painting", contents=contents)
        d3 = PACSGeneralValidation(size=size, keys=["image"], domain="cartoon", contents=contents)
        d4 = PACSGeneralValidation(size=size, keys=["image"], domain="sketch", contents=contents)
        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example

if __name__ == "__main__":

    dt = PACSTrain(size=256)
    print("size PACSTrain:", len(dt))
    dv = PACSValidation(size=256)
    print("size PACSValidation:", len(dv))
    x = dt[0]["image"]
    print("image shape:", x.shape)
    print("dtype of image:", type(x))
    print("max and min in image:", x.max(), x.min())
    print("Data entries:")
    for k in dt[0]:
        print("    ", k, " : ", dt[0][k])
