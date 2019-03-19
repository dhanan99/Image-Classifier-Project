import argparse
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import time
#import workspace_utils 
from PIL import Image
import seaborn as sns
import json    
import train
def process_image(image):
    img=Image.open(image)
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img
def predict(args):
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if ((torch.cuda.is_available() )and (args.device == "cuda")) else "cpu")
    image_path=args.input
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img=img.to("cuda")
    img_torch=img.unsqueeze(0)
    model=train.load_checkpoint(args.checkpoint)
    model.to("cuda")
    model.eval()
    probs=torch.exp(model.forward(img_torch))
    top_probs,top_labels=probs.topk(5,dim=1)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = top_labels.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_probs=top_probs.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_class = [str(idx_to_class[index]) for index in top_labels]
    if args.category_names :
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_flowers=[cat_to_name[label] for label in top_class]
        print(top_flowers)
        print(top_probs)
    else:
        print(top_class)
        print(top_probs)

def main():
    parser = argparse.ArgumentParser(description="This will predict the output using the trained network.")
    parser.add_argument("--input", type = str, dest = "input", help = "The path to the image directory",                                                            default="./flowers/test/10/image_07117.jpg")
    parser.add_argument("--checkpoint", type = str, dest = "checkpoint", default ="checkpoint.pth", help= "Choose the model checkpoint to                                predict")
    parser.add_argument("--category_names",  dest = "category_names", required= False,
                        help="Mapping labels to flower names")
    parser.add_argument("--gpu",  dest = "device", default = 'cuda')   
    args = parser.parse_args()
    predict(args)
   
    
if __name__ == "__main__" :
    main()
    