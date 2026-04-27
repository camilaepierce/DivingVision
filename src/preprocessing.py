import tarfile

#################################
### Assumed Globals
#################################
RGB_TAR = '../data/Diving48_rgb.tar.gz'
FLOW_TAR = '../data/Diving48_flow.tar.gz'


def extract_rgb(rgb_tar=RGB_TAR):
    '''
    Extracts data from RGB data stream. Assumes extracting Diving48 dataset if none given.
    '''
    return tarfile.open(rgb_tar)

def extract_flow(flow_tar=FLOW_TAR):
    '''
    Extracts data from Flow data stream. Assumes extracting Diving48 dataset if none given.
    '''
    return tarfile.open(flow_tar)


def main():
    # TODO: Check if tarfile exists 
    print("Running preprocessor...\n")
    print("Extracting RGB")
    rgb_tarfile = extract_rgb()
    print("Finished RGB extraction, converting to Pytorch tensor")
    ### 
    print("Extracting Flow")
    flow_tarfile = extract_flow()
    print("Finished Flow extraction")
    print(rgb_tarfile.getmembers())

if __name__ == "__main__":
    main()
