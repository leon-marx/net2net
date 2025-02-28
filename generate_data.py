import matplotlib.pyplot as plt
from net2net.models.flows.flow import Net2NetFlow
from net2net.data.pacs import PACSTrain, PACSValidation
import torch
from torch.utils.data import DataLoader


# Hyperparameters
NUM_CLASSES = 3
AE_CKPT_PATH = "logs/2021-11-30T22-23-36_512_pretrained_continue/checkpoints/epoch=000083.ckpt"
CINN_CKPT_PATH = "logs/2021-12-07T23-57-27_512_net2net_3/checkpoints/epoch=000526.ckpt"
DOMAINS = ["photo", "art_painting", "sketch"]
CONTENTS = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
BATCH_SIZE = 16
HOME_PATH = "/home/tarkus/leon/bachelor/"
DATA_PATH = "/data/PACS_augmentations/PAC_train"


def get_model():
    flow_config = {
        "target": "net2net.modules.flow.flatflow.ConditionalFlatCouplingFlow",
        "params": {
            "conditioning_dim": NUM_CLASSES,
            "embedding_dim": 10,
            "conditioning_depth": 2,
            "n_flows": 20,
            "in_channels": 512,
            "hidden_dim": 1024,
            "hidden_depth": 2,
            "activation": "lrelu",
            "conditioner_use_bn": True
        }
    }
    cond_stage_config = {
        "target": "net2net.modules.labels.model.Labelator",
        "params": {
            "num_classes": NUM_CLASSES,
            "as_one_hot": True
        }
    }
    first_stage_config = {
        "target": "net2net.models.autoencoder.BigAE",
        "params": {
            "ckpt_path": AE_CKPT_PATH,
            "encoder_config": {
                "target": "net2net.modules.autoencoder.encoder.ResnetEncoder",
                "params": {
                    "in_channels": 3,
                    "in_size": 256,
                    "pretrained": False,
                    "type": "resnet101",
                    "z_dim": 512
                }
            },
            "decoder_config": {
                "target": "net2net.modules.autoencoder.decoder.BigGANDecoderWrapper",
                "params": {
                    "z_dim": 512,
                    "in_size": 256,
                    "use_actnorm_in_dec": True
                }
            },
            "loss_config": {
                "target": "net2net.modules.autoencoder.loss.DummyLoss"
            }
        }
    }
    ckpt_path = CINN_CKPT_PATH
    N = Net2NetFlow(flow_config=flow_config,
                        first_stage_config=first_stage_config,
                        cond_stage_config=cond_stage_config,
                        ckpt_path=ckpt_path,
                        first_stage_key="image",
                        cond_stage_key="class")
    N.eval()
    return N

def get_dset(mode="train"):
    if mode == "val":
        D = PACSValidation(DOMAINS, CONTENTS)
    elif mode == "train":
        D = PACSTrain(DOMAINS, CONTENTS)
    else:
        print("mode not supported. Choose between 'train' and 'val'.")
    return D

def translation(model, batch):
    cond_dict = {
        0: "photo",
        1: "art_painting",
        2: "cartoon",
        3: "sketch"
    }
    true_conditions = batch["class"]
    with torch.no_grad():
        tc = torch.LongTensor([true_conditions])
        tc = model.cond_stage_model.make_one_hot(tc)
        z = model.first_stage_model.encode(batch["image"], return_mode=True)
        zz, _ = model.flow(z, tc)
        translations = []
        for i in range(4):
            wc = torch.LongTensor([i])
            wc = model.cond_stage_model.make_one_hot(wc)
            z_inv_rec = model.flow.reverse(zz, wc)
            x_inv_rec = model.decode_to_img(z_inv_rec)
            translations.append({"img": x_inv_rec, "cond": cond_dict[i]})
    return translations

def main():
    model = get_model()
    dset = get_dset(mode="train")
    dataloader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False)
    for i, batch in enumerate(dataloader):
        translations = translation(model=model, batch=batch)
        for j, t in enumerate(translations):
            # Check if t has correct shape and values
            # save t to aug_{i}_{j}.jpg
            print("dummy statement")
            

# Execution
if __name__ == "__main__":
    main()


"""

def get_images(dset, size):
    inds = torch.randint(low=0, high=len(dset), size=(size,))
    images = []
    for ind in inds:
        image = dset[ind]
        image["image"] = torch.Tensor(image["image"]).view(1, 256, 256, 3).permute(0, 3, 1, 2)
        images.append(image)
    return images

def make_plot(model, images, save_to=None):
    plt.figure(figsize=(15, 15))
    for i, image in enumerate(images):
        plt.subplot(5, 5, 5*i+1)
        plt.title(f"Original: {image['fname']}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow((image["image"].view(3, 256, 256).permute(1, 2, 0).numpy() + 1) / 2)
        translations = translation(model, image)
        for j in range(4):
            x = translations[j]["img"]
            plt.subplot(5, 5, 5*i+j+2)
            plt.title(f"Condition: {translations[j]['cond']}")
            plt.xticks([])
            plt.yticks([])
            plt.imshow((x.view(3, 256, 256).permute(1, 2, 0).numpy() + 1) / 2)
    if save_to == None:
        plt.show()
    else:
        plt.savefig(save_to)

def make_overview(iterations=1):
    m = get_model()
    d = get_dset()
    for i in range(iterations):
        print(f"Iteration: {i}")
        images = get_images(d, 5)
        make_plot(m, images, save_to=f"/home/tarkus/leon/bachelor/logs/generated_net2net_2/img_{i}.jpg")

"""