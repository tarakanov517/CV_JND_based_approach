import torch
import matplotlib.pyplot as plt
from torch import nn
from evals import evaluate
from model import get_omega_for_parameter


def train(
    net,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    omega1,
    omega2,
    omega3,
    relative_params=True,
):
    losses = []
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True) #
            target = target.to(device, non_blocking=True) #

            optimizer.zero_grad()

            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        omega = get_omega_for_parameter(name, omega1, omega2, omega3)
                        if omega > 0:
                            noise = torch.randn_like(param)

                            if relative_params:
                                scale = param.detach().std()
                                param.add_(omega * scale * noise)
                            else:
                                param.add_(omega * noise)

            running_loss += loss.item()
        if epoch == num_epochs - 1:
            val_acc, val_loss = evaluate(
                model=net,
                data_loader=test_loader,
                device=device,
                criterion=criterion,
                final=True,
            )
        else:
            val_acc, val_loss = evaluate(
                model=net,
                data_loader=test_loader,
                device=device,
                criterion=criterion,
            )
        losses.append(running_loss / len(train_loader))
        print(
            f"epoch: {epoch+1}, train loss: {running_loss/len(train_loader)}, val_acc: {val_acc}, val_loss: {val_loss}"
        )

        plt.plot(losses)
        plt.xlabel("Эпоха")
        plt.ylabel("Loss")
