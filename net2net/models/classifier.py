import os
from net2net.modules.classifier.autoencoder import AE
import pytorch_lightning as pl
import torch
import torch.nn as nn


class Classifier(pl.LightningModule):
    def __init__(self, input_height, latent_dim, ckpt_path=None):
        super().__init__()
        self.repo_path = self.get_repo_path()
        self.ae = AE(input_height=input_height, latent_dim=latent_dim).load_from_checkpoint(self.repo_path + "logs/downloaded/cifar10_resnet18_epoch=96.ckpt")
        self.linear = nn.Linear(self.ae.latent_dim, 7)
        self.loss = nn.CrossEntropyLoss

        if ckpt_path is not None:
            print(f"Loading model from {ckpt_path}")
            self.init_from_ckpt(ckpt_path)

    def get_repo_path(self):
        if os.name == "nt":
            return "C:/users/gooog/desktop/bachelor/code/bachelor/"
        else:
            return "/home/tarkus/leon/bachelor/"

    def init_from_ckpt(self, ckpt_path, ignore_keys=list()):
        try:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        except KeyError:
            sd = torch.load(ckpt_path, map_location="cpu")

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"Missing keys in state dict: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys in state dict: {unexpected}")

    def forward(self, x):
        feats = self.ae.encoder(torch.Tensor(x).view(1, 256, 256, 3).permute(0, 3, 1, 2))
        z = self.ae.fc(feats)
        pred = self.linear(z)
        return pred

    def training_step(self, batch, batch_idx):
        inputs = batch["image"].permute(0, 3, 1, 2)
        labels = batch["content"]
        predictions = self(inputs)
        loss = self.loss(predictions, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"].permute(0, 3, 1, 2)
        labels = batch["content"]
        predictions = self(inputs)
        loss = self.loss(predictions, labels)
        output = pl.EvalResult(checkpoint_on=loss)
        return output

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.ae.parameters())+list(self.linear.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from net2net.data.pacs import PACSValidation
    d = PACSValidation()
    ind = torch.randint(low=0, high=len(d), size=(1,))
    image = d[ind]

    print(image["class"])
    print(image["fname"])
    print(image["content"])
    c = Classifier(256, 256)
    pred = c(image)
    print(pred)
    print("Done")
