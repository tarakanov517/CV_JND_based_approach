import torch


def get_axon_omega(name, omega1, omega2, omega3):
    if name.startswith("block1"):
        return omega1
    if name.startswith("block2"):
        return omega2
    return omega3


def get_dendrite_parameters(
    name,
    theta1,
    sigma1,
    theta2,
    sigma2,
    theta3,
    sigma3,
):
    if name.startswith("block1"):
        return theta1, sigma1
    if name.startswith("block2"):
        return theta2, sigma2
    return theta3, sigma3


def sample_axon_noise(parameter, omega):
    scale = parameter.detach().std(unbiased=False)
    return torch.randn_like(parameter) * scale * omega


class DendriteNoise:
    def __init__(self):
        self.state = {}

    def sample(self, name, parameter, theta, sigma):
        state = self.state.get(name)
        if state is None:
            state = torch.zeros_like(parameter)

        state = state - theta * state + sigma * torch.randn_like(parameter)
        self.state[name] = state

        scale = parameter.detach().std(unbiased=False)
        return state * scale


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    device,
    omega1=0.0,
    omega2=0.0,
    omega3=0.0,
    dendrite_theta1=0.0,
    dendrite_sigma1=0.0,
    dendrite_theta2=0.0,
    dendrite_sigma2=0.0,
    dendrite_theta3=0.0,
    dendrite_sigma3=0.0,
    dendrite_noise=None,
):
    running_loss = 0.0
    mean_accuracy = 0.0

    model.train()

    if dendrite_noise is None:
        dendrite_noise = DendriteNoise()

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        perturbations = []

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue

                omega = get_axon_omega(name, omega1, omega2, omega3)
                theta, sigma = get_dendrite_parameters(
                    name,
                    dendrite_theta1,
                    dendrite_sigma1,
                    dendrite_theta2,
                    dendrite_sigma2,
                    dendrite_theta3,
                    dendrite_sigma3,
                )

                if omega == 0 and sigma == 0 and name not in dendrite_noise.state:
                    continue

                perturbation = torch.zeros_like(parameter)

                if omega != 0:
                    perturbation.add_(sample_axon_noise(parameter, omega))

                if sigma != 0 or name in dendrite_noise.state:
                    perturbation.add_(
                        dendrite_noise.sample(
                            name,
                            parameter,
                            theta,
                            sigma,
                        )
                    )

                parameter.add_(perturbation)
                perturbations.append((parameter, perturbation))

        try:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        finally:
            with torch.no_grad():
                for parameter, perturbation in perturbations:
                    parameter.sub_(perturbation)

        optimizer.step()

        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == targets).float().mean()
        mean_accuracy += accuracy.item()

    running_loss /= len(train_loader)
    mean_accuracy /= len(train_loader)

    return running_loss, mean_accuracy
