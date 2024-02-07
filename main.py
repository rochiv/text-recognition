import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image


class SimpleNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super(SimpleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
        )

        self.adap_avg_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adap_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# Testing the CNN
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')


# Using a custom image
def predict_with_custom_image(model, device, image_path):
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
    # Transform for the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Load MNIST dataset
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    # Checking for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    another_model = SimpleNet().to(device)

    # Optimizer
    optimizer = optim.SGD(another_model.parameters(), lr=0.1, momentum=0.5)

    # Number of epochs
    epochs = 10

    # Training and testing
    for epoch in range(1, epochs + 1):
        train(another_model, device, trainloader, optimizer, epoch)
        test(another_model, device, testloader)

    model_path = "best_model.pth"
    torch.save(another_model.state_dict(), model_path)

    # Predict with custom image (example path)
    predict_with_custom_image(another_model, device, 'test_image_path.png')


if __name__ == '__main__':
    main()

    # load existing model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SimpleNet().to(device)
    #
    # predict_with_custom_image(model, device, image_path="test_eight.png")
