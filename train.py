from modules.args import device, std, mean, TRAIN_PATH, MODELS_PATH, beta_, TEST_PATH, AUDIO_PATH, META_PATH
from modules.model import Net
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from geoopt.optim import RiemannianAdam
from torch.optim import lr_scheduler, Adam
from modules.utils import customized_loss, reveal_loss
import numpy as np
import os
import argparse
import logging  
from modules.test import run_test
import pandas as pd
import numpy as np
import random
random.seed(42)
from modules.esc50 import ESC50Data
import warnings
warnings.filterwarnings("ignore")
import time

logging.basicConfig(filename='prim.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger() 
logger.setLevel(logging.DEBUG) 

def train_model(train_loader, beta, learning_rate, test_loader, n_epochs_stop=20):

    if not os.path.exists(MODELS_PATH): os.mkdir(MODELS_PATH)

    # Save optimizer
    optimizer = RiemannianAdam(net.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5, verbose=True)

    reveal_net_optimizer = RiemannianAdam(net.reveal.parameters(), lr=learning_rate)
    reveal_net_scheduler = lr_scheduler.ReduceLROnPlateau(reveal_net_optimizer, 'min', 0.5, 5, verbose=True)

    loss_history = []
    reveal_loss_history = []
    min_test_loss = 9999999999
    epochs_no_improve = 0
    # Iterate over batches performing forward and backward passes
    for epoch in range(0, num_epochs):
        # Train mode
        net.train()
        
        train_losses = []
        reveal_losses = []
        logger.info(f"Epoch {epoch+1} -> LR : {optimizer.param_groups[0]['lr']}")
        logger.info(f"Epoch {epoch+1} -> LR reveal : {reveal_net_optimizer.param_groups[0]['lr']}")
        print(f"Epoch {epoch+1} -> LR : {optimizer.param_groups[0]['lr']}")
        print(f"Epoch {epoch+1} -> LR reveal : {reveal_net_optimizer.param_groups[0]['lr']}")
        # Train one epoch
        n_mini_batch = len(train_loader)
        for idx, train_batch in enumerate(train_loader):
            print(f"Mini-batch: {idx}/{n_mini_batch}")
            data, _  = train_batch

            # Saves secret images and secret covers
            c = data[:len(data)//2]
            s = data[len(data)//2:]
            
            # Creates variable from secret and cover images
            s = s.to(device)
            c = c.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            c_prime, c_prime_noise, s_prime = net(s, c)
            # Calculate loss and perform backprop
            train_loss, train_loss_cover, train_loss_secret = customized_loss(s_prime, c_prime, s, c, beta) # S', C', S, C
            train_loss.backward(retain_graph=True)

            reveal_net_optimizer.zero_grad()
            s_prime = net.reveal(c_prime_noise)
            reveal_loss_secret = reveal_loss(s_prime, s)      
            reveal_loss_secret.backward()

            optimizer.step()
            reveal_net_optimizer.step()

            # Saves training loss
            train_losses.append(train_loss.item())
            reveal_losses.append(reveal_loss_secret.item())
            loss_history.append(train_loss.item())
            reveal_loss_history.append(reveal_loss_secret.item())
                
            # Prints mini-batch losses
            # print('Training: Batch {0}/{1}. Loss of {2:.4f}, cover loss of {3:.4f}, secret loss of {4:.4f}'.format(idx+1, len(train_loader), train_loss.item(), train_loss_cover.item(), train_loss_secret.item()))

        # Prep model for evaluation

        torch.save(net.state_dict(), MODELS_PATH+f'Epoch N{epoch+1}.pkl')

        mean_train_loss = np.mean(train_losses)
        mean_reveal_loss = np.mean(reveal_losses)
        mean_test_loss = run_test(epoch+1, test_loader)

        scheduler.step(mean_train_loss)
        reveal_net_scheduler.step(mean_reveal_loss)

        logger.info('Epoch [{0}/{1}], Average_train_loss: {2:.4f}'.format(epoch+1, num_epochs, mean_train_loss))
        logger.info('Epoch [{0}/{1}], Average_reveal_loss: {2:.4f}'.format(epoch+1, num_epochs, mean_reveal_loss))
        logger.info('Epoch [{0}/{1}], Average_test_loss: {2:.4f}'.format(epoch+1, num_epochs, mean_test_loss))
        # Prints epoch average loss
        # print('Epoch [{0}/{1}], Average_train_loss: {2:.4f}'.format(epoch+1, num_epochs, mean_train_loss))
        # print('Epoch [{0}/{1}], Average_reveal_loss: {2:.4f}'.format(epoch+1, num_epochs, mean_reveal_loss))
        # print('Epoch [{0}/{1}], Average_test_loss: {2:.4f}'.format(epoch+1, num_epochs, mean_test_loss))

        if mean_test_loss < min_test_loss:
            epochs_no_improve = 0
            min_test_loss = mean_test_loss
        else:
            epochs_no_improve += 1

        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            logger.info('Early stopping!' )
            print('Early stopping!' )
            early_stop = True
            break

    
    return net, mean_train_loss, loss_history, reveal_loss_history

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float,
        help="Learning rate to use for training (default: 0.0001)",
    )

    parser.add_argument(
        "--bs",
        default=64,
        type=int,
        help="Batch size to use for training (default: 64)",
    )

    parser.add_argument(
        "--ep",
        default=100,
        type=int,
        help="No. of epochs for training (default: 100)",
    )

    parser.add_argument(
        "--dataset",
        default="ImageNet",
        type=str,
        help="Dataset either ImageNet or ESC50",
    )

    args = parser.parse_args()

    learning_rate = args.lr
    num_epochs = args.ep
    batch_size = args.bs
    dataset = args.dataset

    net = Net()
    model= torch.nn.DataParallel(net)
    net.to(device)

    if dataset == "ESC50":
        # Creates training set for ESC50
        df = pd.read_csv(f'{META_PATH}/esc50.csv')
        train = df[df['fold']!=5]
        test= df[df['fold']==5]
        train_data = ESC50Data(AUDIO_PATH, train, 'filename', 'category')
        test_data = ESC50Data(AUDIO_PATH, test, 'filename', 'category')

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    else: 
        # Creates training set for ImageNet
        train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                TRAIN_PATH,
                transforms.Compose([
                # transforms.Resize(256),
                # transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                std=std)
                ])), batch_size=batch_size, num_workers=0, 
                pin_memory=True, shuffle=True, drop_last=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
            TEST_PATH, 
            transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
            std=std)
            ])), batch_size=batch_size, num_workers=0, 
            pin_memory=True, shuffle=True, drop_last=True)
    start_time = time.time()
    net, mean_train_loss, loss_history, reveal_loss_history = train_model(train_loader, beta_, learning_rate, test_loader)
    logger.info(f"--- {(time.time() - start_time)} seconds ---")