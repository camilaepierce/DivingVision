# DivingVision

## How to Use

Can be used for training the existing model, creating a new model, etc.

### Virtual Environment

Create a virtual environment

`python3 -m venv ./venv`
`source ./venv/bin/activate`

Install required packages to virtual environment.

`pip install -r requirements.txt`


### Training

To train an existing model, run pipeline.py with the model name as an argument:

`python3 pipeline.py simple`

### Model Creation

Create a model, place it in the `models/` folder under a subfolder with the exact name. Additionally create a config file (template config in `models/`), include `test_split.json` and `train_split.json`.

### Visualization

Training progression is saved automatically after each run as `results/training_history.png`. The plot shows training loss across epochs and, when a test loader is available, test accuracy over the same epochs. Visualization also shows a confusion matrix and a comparison of accuracy by dive class.

## Config Files