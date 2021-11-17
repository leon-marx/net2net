import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

from net2net.data.base import ConcatDatasetWithIndex
import net2net.data.utils as ndu


class PACSDataset(Dataset):
    def __init__(self, domain, contents, train, augment):
        super(PACSDataset, self).__init__()
        if os.name == "nt":
            self.repo_path = f"C:/Users/gooog/Desktop/Bachelor/Code/bachelor/"
        else:
            self.repo_path = f"/home/tarkus/leon/bachelor/"
        self.domain = domain
        self.contents = contents
        self.train = train
        self.augment = augment
        self.pacs_dir = "PACS_train" if self.train else "PACS_test"
        self._prepare()
        self._load()

    def _prepare(self):
        self.root = self.repo_path + f"data/{self.pacs_dir}/{self.domain}/"
        for content in self.contents:
            if not ndu.is_prepared(self.root + f"{content}/"):
                print(f"preparing {self.domain} {content} dataset...")
                # prep
                ndu.mark_prepared(self.root + f"{content}/")
        self._data_path = self.repo_path + f"data/{self.pacs_dir}/meta/{self.domain}.txt"

    def _load(self):
        self._data = np.loadtxt(self._data_path, dtype=str)
        self.labels = {
            "fname": self._data,
            "domain": np.array([str(self.domain)] * self._data.shape[0]),
            "content":self._contents_one_hot(np.vectorize(lambda s: s.split("/")[0])(self._data))
        }
        self._length = self.labels["fname"].shape[0]
        if True:
            print("")
            print("Domain:", self.domain)
            print("shape of fname:", self.labels["fname"].shape)
            print("unique domains:", np.unique(self.labels["domain"]))
            print("unique contents:", np.unique(self.labels["content"], axis=0))
            print("")

    def _contents_one_hot(self, array):
        swap = {}
        for i, content in enumerate(self.contents):
            swap[content] = np.zeros(len(self.contents))
            swap[content][i] = 1
        new = np.zeros((len(array), len(self.contents)))
        for i, a in enumerate(array):
            new[i] = swap[a]
        return new

    def __getitem__(self, i):
        example = self._load_example(i)
        self._augment_example(example, augment=self.augment)
        self._preprocess_example(example)
        example["index"] = i
        return example

    def _load_example(self, i):
        example = dict()
        for k in self.labels:
            example[k] = self.labels[k][i]
        example["image"] = Image.open(os.path.join(self.root, example["fname"]))
        if not example["image"].mode == "RGB":
            example["image"] = example["image"].convert("RGB")
        example["image"] = np.array(example["image"])
        return example

    def _augment_example(self, example, augment):
        # Numbers mean, how good of an idea I think this is. 1 = very good, 5 = maybe catastrophal
        if augment:
            transform = A.Compose([
                A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR, p=1), # 1
                A.RandomResizedCrop(height=256, width=256, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), interpolation=cv2.INTER_LINEAR, p=0.8), # 1
                A.HorizontalFlip(p=0.5), # 1
                A.Rotate(limit=45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=1, p=0.5), # 1
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5), # 1
                A.OneOf([
                    A.Blur(blur_limit=(3, 7), p=0.5), # 2
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.5), # 2
                    A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, mode="fast", p=0.5), # 3
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5), # 2
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5), # 2
                ]),
                A.OneOf([
                    A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.1), # 4
                    A.ChannelShuffle(p=0.1), # 4
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3), # 2
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2), # 3
                    A.Equalize(mode="cv", by_channels=True, p=0.2), # 3
                    A.FancyPCA(alpha=0.1, p=0.3), # 2
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3), # 3
                ]),
            ])
        else:
            transform = A.Compose([
                A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR, p=1)
            ])
        transformed = transform(image=example["image"])
        example["image"] = transformed["image"]

    def _preprocess_example(self, example):
        example["image"] = (example["image"] + np.random.random()) / 256.  # dequantization
        example["image"] = (255 * example["image"])
        example["image"] = example["image"] / 127.5 - 1.0
        example["image"] = example["image"].astype(np.float32)

    def __len__(self):
        return self._length


class PACSTrain(Dataset):
    """
    domain: photo, art_painting, cartoon, sketch
    content: dog, elephant, giraffe, guitar, horse, house, person
    """
    def __init__(self):
        contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        d1 = PACSDataset(domain="photo", contents=contents, train=True, augment=True)
        d2 = PACSDataset(domain="art_painting", contents=contents, train=True, augment=True)
        d3 = PACSDataset(domain="cartoon", contents=contents, train=True, augment=True)
        d4 = PACSDataset(domain="sketch", contents=contents, train=True, augment=True)
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
    def __init__(self):
        contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        d1 = PACSDataset(domain="photo", contents=contents, train=False, augment=False)
        d2 = PACSDataset(domain="art_painting", contents=contents, train=False, augment=False)
        d3 = PACSDataset(domain="cartoon", contents=contents, train=False, augment=False)
        d4 = PACSDataset(domain="sketch", contents=contents, train=False, augment=False)
        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example


if __name__ == "__main__":

    dt = PACSTrain()
    print("size PACSTrain:", len(dt))
    dv = PACSValidation()
    print("size PACSValidation:", len(dv))
    x = dt[0]["image"]
    print("image shape:", x.shape)
    print("dtype of image:", type(x))
    print("max and min in image:", x.max(), x.min())
    print("Data entries:")
    """
    train_domains = []
    for d in dt:
        train_domains.append(d["domain"])
    print(np.unique(train_domains, return_counts=True))
    test_domains = []
    for d in dv:
        test_domains.append(d["domain"])
    print(np.unique(test_domains, return_counts=True))
    """
    for k in dt[0]:
        print("    ", k, " : ", dt[0][k])
    while True:
        plt.figure(figsize=(12, 6))
        i = np.random.randint(0, len(dt))
        plt.subplot(1, 2, 1)
        plt.imshow(dt[i]["image"])
        plt.title(dt[i]["fname"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Train Instance")
        j = np.random.randint(0, len(dv))
        plt.subplot(1, 2, 2)
        plt.imshow(dv[j]["image"])
        plt.title(dv[j]["fname"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Test Instance")
        plt.show()