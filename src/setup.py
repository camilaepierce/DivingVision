import requests
import tarfile

urlRGB = "https://nextcloud.nrp-nautilus.io/public.php/dav/files/eqKMRFHqNCrP77L/?accept=zip"
urlFLOW = "https://nextcloud.nrp-nautilus.io/public.php/dav/files/iGd9k9kYHktNMg4/?accept=zip"
target_path = "Diving48_rgb.tar.gz"

if __name__ == "__main__":
    # TODO: Check if tarfile exists within 

    # Download the file
    response = requests.get(url, stream=True)
    with open(target_path, "wb") as f:
        f.write(response.raw.read())

    # Extract the file
    with tarfile.open(target_path) as tar:
        tar.extractall(path="./extracted_folder")