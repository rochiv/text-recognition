import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from models import SimpleNet
from tqdm import tqdm
from whiteboard import WhiteboardApp
import tk


def train(model, device, train_loader, optimizer, n_epochs, loss_function):
    model.train()
    total_loss = 0
    with tqdm(total=len(train_loader)) as progress:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                progress.set_description(f'Train Epoch: {n_epochs}, Loss: {total_loss / (batch_idx + 1):.6f}')
                progress.update(100)
    progress.close()


# Testing the CNN
def test(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(), tqdm(total=len(test_loader)) as progress_bar:
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            progress_bar.update(1)

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * (correct / len(test_loader.dataset))
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')


# Using a custom image
def predict_with_custom_image(model, device, image_path=None):
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
    print(f'Predicted Digit: {pred.item()}')


# Main block
def main():
    # Checking for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform for the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Load MNIST dataset class
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # model instance
    model = SimpleNet().to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Number of epochs
    epochs = 20

    # Loss function
    loss = nn.CrossEntropyLoss()

    # Training and testing
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        test(model, device, test_loader, loss)

    model_path = "best_model.pth"
    torch.save(model.state_dict(), model_path)

    # Predict with custom image (example path)
    predict_with_custom_image(model, device, 'test_one.png')


if __name__ == '__main__':
    main()

    # load existing model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SimpleNet().to(device)
    #
    # predict_with_custom_image(model, device, image_path="test_eight.png")
