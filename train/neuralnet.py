import torch
import torch.nn as nn
import torch.optim as optim

# NeuralNet class encapsulates the SRNET model and its operations
class NeuralNet(object):
    def __init__(self, device, ngpu):
        super(NeuralNet, self).__init__()
        self.device = device
        self.ngpu = ngpu

        # Initialize the SRNET model
        self.model = SRNET(self.ngpu).to(self.device)

        # If multiple GPUs are available, use DataParallel
        if (self.device.type == 'cuda') and (self.ngpu > 0):
            self.model = nn.DataParallel(self.model, list(range(self.ngpu)))

        # Print model summary and parameter count
        num_params = sum(p.numel() for p in self.model.parameters())
        print(self.model)
        print(f"The number of parameters: {num_params}")

        # Loss function and optimizer
        self.mse = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)

    # Save the model's state_dict
    def save(self, path):
        torch.save(self.model.state_dict(), path)  # Save only the model parameters
        print(f"Model saved to {path}")

    # Load the model's state_dict
    def load(self, path):
        self.model.load_state_dict(torch.load(path))  # Load only the model parameters
        self.model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {path}")

# SRNET class defines the structure of the neural network model
class SRNET(nn.Module):
    def __init__(self, ngpu):
        super(SRNET, self).__init__()

        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=9//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=5//2),
        )

    def forward(self, input):
        # Apply the sequential layers and clamp the output to avoid extreme values
        return torch.clamp(self.model(input), min=1e-12, max=1-(1e-12))
