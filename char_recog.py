import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from IPython import display
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from cnn_model import cnn_model
display.set_matplotlib_formats("svg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# * DOWNLOAD THE DATASET
cdata = torchvision.datasets.EMNIST(
    root="emnist", split="letters", download=True)
char_classes = cdata.classes[1:]  # Remove 'N/A' category
char_labels = cdata.targets - 1

print(f"NUMBER OF TRAINING SAMPLES: {len(cdata)}")
print(f"NUMBER OF CLASSES: {len(char_classes)}")
print(f"CLASSES: {char_classes}")
print(f"DATA SIZE: {cdata.data.shape}\n")

char_images = cdata.data.view([len(cdata), 1, 28, 28]).float()

# * NORMALIZE IMAGES
char_images /= torch.max(char_images)

# * CREATE TRAINING AND TESTING DATA
train_char_data, test_char_data, train_char_labels, test_char_labels = train_test_split(
    char_images, char_labels, test_size=.1)

# * CONVERT INTO PYTORCH DATASETS
train_char_data = TensorDataset(train_char_data, train_char_labels)
test_char_data = TensorDataset(test_char_data, test_char_labels)

# * TRANSLATE INTO DATA LOADER OBJECTS
train_char_loader = DataLoader(
    train_char_data, batch_size=32, shuffle=True, drop_last=True)
test_char_loader = DataLoader(
    test_char_data, batch_size=test_char_data.tensors[0].shape[0], shuffle=True, drop_last=True)


def train_model():
    num_epochs = 10
    net, lossfun, optimizer = cnn_model(output_layers=26)
    # Send model to GPU
    net.to(device)

    trainLoss = torch.zeros(num_epochs)
    testLoss = torch.zeros(num_epochs)

    trainErr = torch.zeros(num_epochs)
    testErr = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        net.train()
        batchLoss = []
        batchErr = []

        # Forward and backward propagate for digits
        for x, y in train_char_loader:
            # Push data to GPU
            x = x.to(device)
            y = y.to(device)

            # Forward progagate and calculate loss
            yHat = net(x)
            loss = lossfun(yHat, y)

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Append loss and error from this batch
            batchLoss.append(loss.item())
            batchErr.append(torch.mean(
                (torch.argmax(yHat, axis=1) != y).float()).item())

        # Get average losses and error rates across the batches
        trainLoss[epoch] = np.mean(batchLoss)
        print(f"TRAIN LOSS: {trainLoss[epoch]}")
        trainErr[epoch] = 100 * np.mean(batchErr)
        print(f"TRAIN ERR: {trainErr[epoch]}")

        # * TEST PERFORMANCE
        net.eval()
        xChar, yChar = next(iter(test_char_loader))
        xChar = xChar.to(device)
        yChar = yChar.to(device)

        with torch.no_grad():
            yHatChar = net(xChar)
            lossChar = lossfun(yHatChar, yChar)

        # Get loss and error rate from the test
        testLoss[epoch] = lossChar.item()
        print(f"TEST LOSS CHAR: {testLoss[epoch]}")
        testErr[epoch] = 100 * \
            torch.mean((torch.argmax(yHatChar, axis=1)
                       != yChar).float()).item()
        print(f"TEST ERR CHAR: {testErr[epoch]}")

    return trainLoss, testLoss, trainErr, testErr, net


# * RUN THE MODEL AND SHOW THE RESULTS
trainLoss, testLoss, trainErr, testErr, net = train_model()
torch.save(net.state_dict(), "HandwrittenCharacterRecognition/char_recog.pth")

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

print(trainLoss, testLoss)

ax[0].plot(trainLoss, "s-", label="Train")
ax[0].plot(testLoss, "o-", label="Test")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss (MSE)")
ax[0].set_title("Model loss")

ax[1].plot(trainErr, "s-", label="Train")
ax[1].plot(testErr, "o-", label="Test")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Error rates (%)")
ax[1].set_title(f"Final model test error rate: {testErr[-1]:.2f}%")
ax[1].legend()

plt.show()
