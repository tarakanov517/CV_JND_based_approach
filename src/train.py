from evals import evaluate


def train(
    net,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
):
    history = []

    for epoch in range(1, num_epochs + 1):
        net.train()
        loss_sum = 0.0
        sample_count = 0

        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * target.size(0)
            sample_count += target.size(0)

        train_loss = loss_sum / sample_count
        val_acc, val_loss = evaluate(
            model=net,
            data_loader=test_loader,
            device=device,
            criterion=criterion,
            final=epoch == num_epochs,
        )
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_accuracy": val_acc,
            "validation_loss": val_loss,
        }
        history.append(metrics)
        print(
            f"epoch: {epoch}, train loss: {train_loss}, "
            f"val_acc: {val_acc}, val_loss: {val_loss}",
            flush=True,
        )

    return history
