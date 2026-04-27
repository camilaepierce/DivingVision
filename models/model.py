from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        ### START TODO: initialize model
        self.layers = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=2, kernel_size=9, stride=1),
          nn.ReLU(),
          nn.AdaptiveAvgPool2d(output_size=(1, 1)),
          nn.Flatten(),
          nn.LazyLinear(out_features=2)
        )

        ### END TODO
    def forward(self, x):
        ### START TODO: complete forward pass of model
        x = self.layers.forward(x)
        ### END TODO
        return x