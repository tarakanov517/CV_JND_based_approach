import torch


def eval(model, test_loader, criterion, device):
    model.eval()
    mean_loss = 0.0
    mean_acc = 0.0
    for X_batch, y_true in test_loader:
        X_batch = X_batch.to(device)
        y_true = y_true.to(device)

        out = model(X_batch)
        loss = criterion(out, y_true)
        mean_loss += loss.item()

        y_pred = torch.argmax(out, 1)
        accuracy = torch.sum(y_pred == y_true) / y_pred.shape[0]
        mean_acc += accuracy.item()
    mean_loss /= len(test_loader)
    mean_acc /= len(test_loader)
    return mean_loss, mean_acc
