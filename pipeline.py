import argparse
import src
from models.preprocessing import main
from src.training import train_model
from src.dataset import DivingDataset
from src.model import SimpleCNN



def main():
    ################################
    ### Setup
    ################################

    # CPU / GPU
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    ################################
    ### Data Processing
    ################################

    rgb_tarfile = extract_rgb()

    # Create datasets and dataloaders for train and validation
    train_dataset = DivingDataset(XTrain, YTrain)
    test_dataset = DivingDataset(XTest, YTest)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    ################################
    ### Training
    ################################

    model = SimpleCNN().to(device)
    model = train_model(model, train_loader, test_loader, device)

    ################################
    ### Evalution
    ################################



    ################################
    ### Visualization
    ################################


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Specify an alternate config file. File must exist and follow the config requirements as described in the README.")
    parser.add_argument("-r", "--results", help="Specify an alternate results folder. Folder will be created if it does not exist.")
    args = parser.parse_args()

    if args.flow:
        print("Using flow")

    if args.rgb:
        print("Using rgb")