import torch
from modules.model import Net
from modules.args import device, MODELS_PATH, TEST_PATH, std, mean, beta_, AUDIO_PATH, META_PATH
from torchvision import datasets, utils
import torchvision.transforms as transforms
from modules.utils import customized_loss, imshow
import numpy as np
import argparse
import pandas as pd
from modules.esc50 import ESC50Data
import warnings
warnings.filterwarnings("ignore")


def run_test(epoch, test_loader):

    trained_model = Net()
    trained_model.load_state_dict(torch.load(MODELS_PATH+f'Epoch N{epoch}.pkl', map_location=device))
    # trained_model.load_state_dict(torch.load(f'Epoch N155.pkl'))

    trained_model.to(device)

    # Switch to evaluate mode
    trained_model.eval()

    # Creates test set

    test_losses = []
    cover_losses = []
    secret_losses = []
    # Show images

    with torch.no_grad():
        for idx, test_batch in enumerate(test_loader):
            # Saves images
            data, _ = test_batch

            # Saves secret images and secret covers
            test_secret = data[:len(data)//2]
            test_cover = data[len(data)//2:]

            # Creates variable from secret and cover images
            test_secret = test_secret.to(device)
            test_cover = test_cover.to(device)

            # Compute output
            test_hidden, _, test_output = trained_model(test_secret, test_cover)
            
            # Calculate loss
            test_loss, loss_cover, loss_secret = customized_loss(test_output, test_hidden, test_secret, test_cover, beta_)
                    
            # print (diff_S, diff_C)
            
            # if idx in [1,2,3,4]:
            #     print ('Total loss: {:.2f} \nLoss on secret: {:.2f} \nLoss on cover: {:.2f}'.format(test_loss.item(), loss_secret.item(), loss_cover.item()))

            #     # Creates img tensor
            #     imgs = [test_secret.data, test_output.data, test_cover.data, test_hidden.data] # S, S', c, C'
            #     imgs_tsor = torch.cat(imgs, 0).to(torch.device('cpu'))

            #     # Prints Images
            #     imshow(utils.make_grid(imgs_tsor), idx+1, learning_rate=0.01, beta=beta_)
                
            test_losses.append(test_loss.item())
            cover_losses.append(loss_cover.item())
            secret_losses.append(loss_secret.item())
            
    mean_test_loss = np.mean(test_losses)
    mean_cover_loss = np.mean(cover_losses)
    mean_secret_loss = np.mean(secret_losses)

    print ('Average loss on test set: {:.2f}'.format(mean_test_loss))
    print ('Average loss on test set cover: {:.2f}'.format(mean_cover_loss))
    print ('Average loss on test set secret: {:.2f}'.format(mean_secret_loss))

    return mean_test_loss

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bs",
        default=64,
        type=int,
        help="Batch size to use for training (default: 64)",
    )

    parser.add_argument(
        "--ep",
        default=193,
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

    epoch = args.ep
    batch_size = args.bs
    dataset = args.dataset

    if dataset == "ESC50":
        # Creates training set for ESC50
        df = pd.read_csv(f'{META_PATH}/esc50.csv')
        test= df[df['fold']==5]
        test_data = ESC50Data(AUDIO_PATH, test, 'filename', 'category')

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    else:
        
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
            TEST_PATH, 
            transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
            std=std)
            ])), batch_size=batch_size, num_workers=1, 
            pin_memory=True, shuffle=True, drop_last=True)

    run_test(epoch, test_loader)