import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

from net2net.data.base import ConcatDatasetWithIndex
import net2net.data.utils as ndu


class ImageNetDataset(Dataset):
    def __init__(self, domain, contents, train, augment):
        super(ImageNetDataset, self).__init__()
        if os.name == "nt":
            self.repo_path = f"C:/Users/gooog/Desktop/Bachelor/Code/bachelor/"
        else:
            self.repo_path = f"/home/tarkus/leon/bachelor/"
        if domain != "photo":
            print("WARNING: domain must be photo for Image-Net!")
        self.domain = domain
        self.contents = contents
        self.train = train
        self.augment = augment
        self.imagenet_dir = "imagenet_train" if self.train else "imagenet_test"
        self._prepare()
        self._load()

    def _prepare(self):
        self.root = self.repo_path + f"data/{self.imagenet_dir}/{self.domain}/"
        for content in self.contents:
            if not ndu.is_prepared(self.root + f"{content}/"):
                print(f"preparing {self.domain} {content} dataset...")
                # prep
                ndu.mark_prepared(self.root + f"{content}/")
        self._data_path = self.repo_path + f"data/{self.imagenet_dir}/meta/{self.domain}.txt"

    def _load(self):
        self._data = np.loadtxt(self._data_path, dtype=str)
        self.labels = {
            "fname": self._data,
            "domain": np.array([str(self.domain)] * self._data.shape[0]),
            "content":self._contents_one_hot(np.vectorize(lambda s: s.split("/")[0])(self._data))
        }
        self._length = self.labels["fname"].shape[0]
        if False:
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
        if (example["image"].shape[0] < 256 and example["image"].shape[1] < 256):
            transformed = A.Resize(height=256, width=256, interpolation=cv2.cv2.INTER_LINEAR, p=1.0)(image=example["image"])
            example["image"] = transformed["image"]
        elif example["image"].shape[0] < 256:
            transformed = A.Resize(height=256, width=example["image"].shape[1], interpolation=cv2.cv2.INTER_LINEAR, p=1.0)(image=example["image"])
            example["image"] = transformed["image"]
        elif example["image"].shape[1] < 256:
            transformed = A.Resize(height=example["image"].shape[0], width=256, interpolation=cv2.cv2.INTER_LINEAR, p=1.0)(image=example["image"])
            example["image"] = transformed["image"]
        if augment:
            transform = A.Compose([
                A.RandomCrop(height=256, width=256, p=1.0), # 1
                A.HorizontalFlip(p=0.5), # 1
                A.Rotate(limit=45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE, p=0.5), # 1
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5), # 1
                A.Blur(blur_limit=(3, 4), p=0.3), # 2
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2), # 2
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2), # 3
                A.FancyPCA(alpha=0.1, p=0.2), # 2
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2), # 3
            ])
        else:
            transform = A.Compose([
                A.RandomCrop(height=256, width=256, p=1.0), # 1
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


class ImageNetTrain(Dataset):
    """
    domain: photo
    content: dog, elephant, giraffe, guitar, horse, house, person
    """
    def __init__(self):
        contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        d1 = ImageNetDataset(domain="photo", contents=contents, train=True, augment=True)
        self.data = ConcatDatasetWithIndex([d1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example


class ImageNetValidation(Dataset):
    """
    domain: photo
    content: dog, elephant, giraffe, guitar, horse, house, person
    """
    def __init__(self):
        contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        d1 = ImageNetDataset(domain="photo", contents=contents, train=False, augment=False)
        self.data = ConcatDatasetWithIndex([d1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example


if __name__ == "__main__":

    class_dict = {
        0: "photo"
    }

    dt = ImageNetTrain()
    print("size ImageNetTrain:", len(dt))
    dv = ImageNetValidation()
    print("size ImageNetValidation:", len(dv))
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
        i = 2944
        print(i)
        plt.subplot(2, 2, 1)
        plt.imshow(dt[i]["image"])
        plt.title(dt[i]["fname"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"{np.round(dt[i]['image'].max(), 2)}, {np.round(dt[i]['image'].min(), 2)}")
        plt.ylabel("Dataloader Image")
        j = np.random.randint(0, len(dv))
        j = 3
        plt.subplot(2, 2, 2)
        plt.imshow(dv[j]["image"])
        plt.title(dv[j]["fname"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"{np.round(dv[j]['image'].max(), 2)}, {np.round(dv[j]['image'].min(), 2)}")
        plt.ylabel("Dataloader Image")
        plt.subplot(2, 2, 3)
        ot_path = "data/imagenet_train/" + class_dict[dt[i]["class"]] + "/" + dt[i]["fname"]
        ot_img = np.asarray(Image.open(ot_path, "r"))
        plt.imshow(ot_img)
        plt.title(dt[i]["fname"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"{ot_img.max()}, {ot_img.min()}")
        plt.ylabel("Original Image")
        plt.subplot(2, 2, 4)
        ov_path = "data/imagenet_test/" + class_dict[dv[j]["class"]] + "/" + dv[j]["fname"]
        ov_img = np.asarray(Image.open(ov_path, "r"))
        plt.imshow(ov_img)
        plt.title(dv[j]["fname"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"{ov_img.max()}, {ov_img.min()}")
        plt.ylabel("Original Image")
        plt.close()
        # plt.show()