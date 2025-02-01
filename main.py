import torch
import torch.nn as neural_network
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


input_dimension = 468  # Amount of mediapipe landmarks

layer_1_size = 64
layer_2_size = 64

output_size = 2


class LandmarkEmotionModel(neural_network.Module):
    def __init__(self):
        super().init()

        # initialize layers
        self.layer1 = neural_network.Linear(input_dimension, layer_1_size)
        self.layer2 = neural_network.Linear(layer_1_size, layer_2_size)
        self.output_layer = neural_network.Linear(layer_2_size, output_size)

        # initialize relu
        self.relu = neural_network.ReLU()

    def forward_prop(self, x):
        # x = vector of input value; values going thru neural network
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))

        # sigmoid for last layer to obtain continuous result
        x = torch.sigmoid(self.output_layer(x))

        return x

# WARN: -- copied straight from gpt --


class LandmarkEmotionDataset(Dataset):
    def init(self, landmarks, labels):
        """
        :param landmarks: List or array of flattened landmarks (shape: num_samples x 1404)
        :param labels: List or array of labels (0 for neutral, 1 for angry) (shape: num_samples)
        """
        self.landmarks = torch.tensor(
            landmarks, dtype=torch.float32)  # Convert to tensor
        # Convert labels to tensor
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def len(self):
        return len(self.landmarks)

    def getitem(self, idx):
        return self.landmarks[idx], self.labels[idx]

# WARN: ------------------------------


model = LandmarkEmotionModel()
criterion = neural_network.BCELoss()  # calculates how wrong answer was

# lr (learning rate) determines how much we will change weights/biases in training
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, num_epochs=10):
    model.train()  # put model in training mode

    # TODO: look up what epoch is
    for epoch in range(num_epochs):
        running_loss = 0.0  # sums up loss for epoch
        for landmarks, labels in train_loader:
            optimizer.zero_grad()  # reset training for each epoch

            # run thru neural network and get output
            outputs = model(landmarks)

            # calculate loss for epoch
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        # log
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


# WARN: -- copied straight from gpt --

def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for landmarks, labels in test_loader:
            outputs = model(landmarks)
            # Threshold at 0.5 for binary classification
            predicted = (outputs.squeeze() > 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")


# WARN: ------------------------------

# run training
test_dataset = LandmarkEmotionDataset(test_landmarks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_model(model, train_loader, num_epochs=10)

evaluate_model(model, test_loader)
