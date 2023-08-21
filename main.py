import os
import json
import torch
import numpy as np
import argparse
import time
import requests
import torchvision

from PIL import Image
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode
from surfree import SurFree

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

def get_model():
    model = torchvision.models.resnet50(pretrained=True).eval()
    # normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
    return model

def get_imagenet_labels():
    response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
    return eval(response.content)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="results_test/", help="Output folder")
    parser.add_argument("--n_images", "-n", type=int, default=2, help="N images attacks")
    parser.add_argument("--start", "-s", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dct_type", type=str, default="full")
    parser.add_argument("--frequence_range", type=float, default=0.5)
    parser.add_argument("--basis_type", type=str, default="dct")
    parser.add_argument(
        "--config_path", 
        default="config_example.json", 
        help="Configuration Path with all the parameter for SurFree. It have to be a dict with the keys init and run."
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ###############################
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        raise ValueError("{} doesn't exist.".format(output_folder))
    
    ###############################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ###############################
    print("Load Model")
    model = get_model()

    ###############################
    print("Load Config")
    if args.config_path is not None:
        if not os.path.exists(args.config_path):
            raise ValueError("{} doesn't exist.".format(args.config_path))
        config = json.load(open(args.config_path, "r"))
    else:
        config = {"init": {}, "run": {"epsilons": None}}

    ###############################
    print("Get understandable ImageNet labels")
    imagenet_labels = get_imagenet_labels()


    ###############################
    print("Load Labels")
    ground_truth  = open(os.path.join('val.txt'), 'r').read().split('\n')
        
    ###############################
    print("Load Data")
    X = []
    # transform = T.Compose([T.Resize((224, 224)), T.CenterCrop(224)])
    for image_i in range(args.start, args.n_images+args.start):
        image_name = format(image_i, '08d')
        ground_name_label = ground_truth[image_i-1]
        ground_label =  ground_name_label.split()[1]
        ground_label_int = int(ground_label)
        x_i = Image.open(os.path.join("./images", f"ILSVRC2012_val_{image_name}.JPEG"))
        x_i = T.Compose([T.Resize((224, 224))])(x_i)
        x_i = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(mean = mean, std = std)])
        x_i = x_i[None, :, :, :]
        y_i = model(x_i).argmax(1)[0]
        if y_i == ground_label_int:
            X.append(x_i)
    X = torch.cat(X, 0)
    print(len(X))
    y = model(X).argmax(1)

    ###############################
    print("Attack !")
    time_start = time.time()

    f_attack = SurFree(**config["init"])

    if torch.cuda.is_available():
        model = model.cuda(0)
        X = X.cuda(0)
        y = y.cuda(0)

    advs, results = f_attack(model, X, y, **config["run"])
    print("{:.2f} s to run".format(time.time() - time_start))
    ###############################
    config['run']['basis_params']['dct_type'] = args.dct_type
    config['run']['basis_params']['frequence_range'][1] = args.frequence_range
    config['run']['basis_params']['basis_type'] = args.basis_type
    dct_v = config['run']['basis_params']['dct_type']
    freq_v = config['run']['basis_params']['frequence_range'][1]
    np.save(f'{args.seed}_{dct_v}_{freq_v}_array.npy', results)


    ###############################
    print("Results")
    labels_advs = model(advs).argmax(1)
    nqueries = f_attack.get_nqueries()
    advs_l2 = (X - advs).flatten(1).norm(dim=1)
    for image_i in range(len(X)):
        print("Adversarial Image {}:".format(image_i))
        label_o = int(y[image_i])
        label_adv = int(labels_advs[image_i])
        print("\t- Original label: {}".format(imagenet_labels[str(label_o)]))
        print("\t- Adversarial label: {}".format(imagenet_labels[str(label_adv)]))
        print("\t- l2 = {}".format(advs_l2[image_i]))
        print("\t- {} queries\n".format(nqueries[image_i])) 