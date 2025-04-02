import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################
class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNetV2, self).__init__()

        #load pretrained model with imagnet weights
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # Freeze early feature extraction layers (Only freezes first 5 blocks)
        for param in self.model.features[:5].parameters():
            param.requires_grad = False # Disable gradient computation
            
        num_features = self.model.classifier[1].in_features #adjusts input for the data 
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), #dropout for regularization 
            nn.Linear(in_features=num_features, out_features=num_classes), #final layer modification to fit dataset
        )
    
    def forward(self, x):
        # forward pass
        return self.model(x)

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() #removes prev batch gradients 

        outputs = model(inputs) 
        loss = criterion(outputs, labels)

        loss.backward() #back prop
        optimizer.step() 

        running_loss += loss.item()   ### calc running loss
        _, predicted = torch.max(outputs, 1)    ### get predicted label 

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) ###  -- inference
            loss = criterion(outputs, labels)    ###  -- loss calculation

            running_loss += loss.item()  ### SOLUTION -- add loss from this sample
            _, predicted = torch.max(outputs.data, 1)   ### SOLUTION -- predict the class

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():

    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.


    CONFIG = {
        "model": "EfficientNetV2",   #model name
        "batch_size": 128, # batch size
        "learning_rate": 3e-4, #L2
        "epochs": 50,  
        "num_workers": 4, 
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood3",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Better match for EfficientNet
        transforms.RandomHorizontalFlip(), #random flip
        transforms.RandomRotation(15), #random rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 stats
                        std=[0.2675, 0.2565, 0.2761])
])

    ###############
    #  Add validation and test transforms - NO augmentation for validation/test
    ###############

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Consistent with training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                        std=[0.2675, 0.2565, 0.2761])
])

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))   ###  -- Calculate training set size
    val_size = len(trainset) - train_size     ###  -- Calculate validation set size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])  ###  -- split into training and validation sets

    ###  -- define loaders and test set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"],
                                             shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"],
                                           shuffle=False, num_workers=CONFIG["num_workers"])

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                            shuffle=False, num_workers=CONFIG["num_workers"])
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = EfficientNetV2(num_classes=100)   # instantiate your model ### TODO
    model = model.to(CONFIG["device"])   # move it to target device

    print("\nModel summary:")
    print(f"{model}\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss()   ###  -- define loss criterion
    ### AdamW optimizer, configures the feature params to have a learning rate 1/10 of the classifier learning rate 
    optimizer = optim.AdamW([
    {'params': model.model.features.parameters(), 'lr': CONFIG["learning_rate"]/10},
    {'params': model.model.classifier.parameters(), 'lr': CONFIG["learning_rate"]}], weight_decay=0.01 ) 
    ### cosing annealing LR optmizer. with a min lr of 1e-5 at the end of annealing 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)  

    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth") # Save to wandb as well

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood3.csv", index=False)
    print("submission_ood3.csv created successfully.")

if __name__ == '__main__':
    main()
