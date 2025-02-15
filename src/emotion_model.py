import torch
import torch.nn as nn

class LandmarkEmotionModel(nn.Module):
    def __init__(self,layer_1_size = 64, layer_2_size = 64):
        super().__init__()

        input_dimension = 956  # Amount of mediapipe landmarks/2
        output_size = 1 #(Angry & Not Angry)

        # initialize layers
        self.layer_1 = nn.Linear(input_dimension, layer_1_size)
        self.layer_2 = nn.Linear(layer_1_size, layer_2_size)
        self.output_layer = nn.Linear(layer_2_size, output_size)

        # initialize relu
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = vector of input value; values going thru neural network
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))

        # sigmoid for last layer to obtain continuous result
        x = self.output_layer(x)

        return x