import torch


def get_omega_for_parameter(name, omega1, omega2, omega3):
    if name.startswith("block1"):
        return omega1
    elif name.startswith("block2"):
        return omega2
    else:
        return omega3


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    device,
    omega1,
    omega2,
    omega3,
):
    running_loss = 0.0
    mean_acc = 0.0

    model.train()

    for X_batch, y_true in train_loader:
        X_batch = X_batch.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()

        perturbations = []

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                omega = get_omega_for_parameter(
                    name,
                    omega1,
                    omega2,
                    omega3,
                )

                if omega <= 0:
                    continue
                scale = param.detach().std(unbiased=False)
                noise = torch.randn_like(param)
                perturbation = omega * scale * noise

                param.add_(perturbation)
                perturbations.append((param, perturbation))

        out = model(X_batch)
        loss = criterion(out, y_true)
        loss.backward()

        with torch.no_grad():
            for param, perturbation in perturbations:
                param.sub_(perturbation)

        optimizer.step()

        running_loss += loss.item()

        y_pred = torch.argmax(out, dim=1)
        accuracy = (y_pred == y_true).float().mean()
        mean_acc += accuracy.item()

    running_loss /= len(train_loader)
    mean_acc /= len(train_loader)

    return running_loss, mean_acc
