import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import time
#import workspace_utils
import matplotlib.pyplot as plt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import seaborn as sns
structure={"vgg16":25088,
           "inception":2048}
def data_processing(path):
    data_dir = path
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    train_transforms=transforms.Compose([transforms.RandomRotation(45),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])
    ])
    test_transforms=transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)
    

    train_datasets = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_datasets = torch.utils.data.DataLoader(test_data, batch_size=64)
                                     
    return train_datasets, test_datasets, train_data
def load_model(struct,hidden_size,hidden_size1):
    if struct not in structure.keys():
        print("model not found")
    else:
        if struct=="vgg16":
          model= models.vgg16(pretrained=True)        
        else:
            model=models.inception(pretrained=True)
            
    for param in model.parameters():
        param.requires_grad = False                                        
    classifier=nn.Sequential(nn.Linear(25088,hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(hidden_size,hidden_size1),
                             nn.ReLU(),
                             nn.Linear(hidden_size1,102),
                             nn.LogSoftmax(dim=1))
            
    model.classifier=classifier
    return model      
def train(argps,):
    path=argps.dir
    train_datasets, test_datasets, train_data = data_processing(path)
    device = torch.device("cuda" if ((torch.cuda.is_available() )and (args.device == "cuda")) else "cpu")
    model = load_model(args.struct,args.hidden_units,args.hidden_units1)  
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    print_every=5
    print_count=0
    epochs=3
    model.to("device")
    for e in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in dataloaders["train"]:
            print_count += 1

                # Put model in right "mood"
            model.train()

                # Sent data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

                # (Re)set gradients to zero 
            optimizer.zero_grad()

                ### Forward Pass ###
            outputs = model.forward(inputs)
                # Measure the loss
            loss = criterion(outputs, labels)

                ### Backpropagation ###
            loss.backward()
                # Optimize weights using optimizer
            optimizer.step()

                # Capture the loss
            running_loss += loss.item()
            if print_count % print_every == 0:
                validation_loss = 0
                accuracy = 0
               # Put model in evaluation "mood"
                model.eval()

                # turn off gradients f
                with torch.no_grad():
                    for inputs, labels in dataloaders["valid"]:
                    # move data to device
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                       # feeding
                        outputs = model.forward(inputs) #logps
                     ### Validation Loss and Accuracy ###
                        validation_loss += criterion(outputs, labels)                    
                        ps = torch.exp(outputs) # exp(outputs) = probabilities
                        top_p, top_class = ps.topk(1, dim=1)                    
                        # check if equal
                        equals = top_class == labels.reshape(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                train_loss = running_loss / len(dataloaders["train"])
                validation_loss = validation_loss / len(dataloaders["valid"])
                validation_accuracy = accuracy / len(dataloaders["valid"])
                train_losses.append(running_loss/len(dataloaders["train"]))

                print("Epoch: {}/{} |".format(e+1, epochs),
                          "Training Loss: {:.3f}|".format(train_loss),
                          "Valid. Loss: {:.3f}|".format(validation_loss),
                          "Valid. Accuracy: {:.3f}".format(validation_accuracy))

                running_loss = 0
                model.train()                                     
    save_checkpoint(model, args, train_data)
                                        
def save_checkpoint(device,model,args,data):
    model.class_to_idx=data.class_to_idx
    checkpoint={"structure":args.struct,
            'model_state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx,
            "epochs":2,
            "input_size":structure[args.struct],
            "output_size":102,
            "hidden_size":args.hidden_units,
            "hidden_size1":args.hidden_units1,
            "lr":0.001,
            "optimizer_state_dict":optimizer.state_dict()
           }
    torch.save(checkpoint,"checkpoint.pth")
def load_checkpoint(path="checkpoint.pth"):
    checkpoint=torch.load(path)
    structure=checkpoint["structure"]
    input_size = checkpoint['input_size']
    hidden_layer = checkpoint['hidden_size']
    hidden_layer1 = checkpoint['hidden_size1']
    output_size = checkpoint['output_size']
    learning_rate= checkpoint['lr']
    model = load_model(structure,hidden_layer,hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])    
    #model.to("cpu")
    return model
def main():
    parser = argparse.ArgumentParser(description="This will train the neural network")
    parser.add_argument("--dir", type = str, dest = "dir", help = "The path to the data directory, enter the parent directory name",                                default="./flowers/")
    parser.add_argument("--save_dir", type = str , dest = "save_dir", help ="Set directory to save checkpoints",
                       action="store", default="./checkpoint.pth")
    parser.add_argument("--learning_rate", type = float, dest = "learning_rate", default = 0.001, help= "Choose the learning rate to train") 
    parser.add_argument("--hidden_units", type = int, dest = "hidden_units", help= "Choose the hidden nodes to train the                                model")
    parser.add_argument("--hidden_units1", type = int, dest = "hidden_units1", help= "Choose the hidden nodes to train the                                model")
    parser.add_argument("--struct", type = str, dest = "struct", default ="vgg16", help= "Choose the architecture to train the model")
    parser.add_argument("--input_size", type = int, dest = "input_size", help= "Choose the hidden nodes to train the model")
    parser.add_argument("--gpu", type=str, dest = "device", default = "cuda")
    args = parser.parse_args()
    print(args)
    
    train(args)

    print("Checkpoint saved to {}".format(args.save_dir))
    model=load_checkpoint(args.save_dir)                                        

if __name__=="main":
    main()                                        
        