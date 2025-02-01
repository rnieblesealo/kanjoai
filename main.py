import torch
import torch.nn as neural_network
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


input_dimension = 468  # Amount of mediapipe landmarks

layer_1_size = 64
layer_2_size = 64


class LandmarkEmotionModel(neural_network.Module):
    def __init__(self):
        super().init()

        self.layer1 = neural_network.Linear(input_dimension, layer_1_size)
