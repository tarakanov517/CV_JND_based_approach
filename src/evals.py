import torch


def evaluate(
    model,
    data_loader,
    device,
    criterion=None,
    final=False,
):
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    correct_pred = {i: 0 for i in range(10)}
    total_pred = {i: 0 for i in range(10)}

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            prediction = output.argmax(dim=1)

            if criterion is not None:
                loss = criterion(output, target)
                running_loss += loss.item()

            correct += prediction.eq(target).sum().item()
            total += target.size(0)

            if final:
                for label, predicted_label in zip(
                    target,
                    prediction,
                ):
                    label = label.item()
                    predicted_label = predicted_label.item()

                    if label == predicted_label:
                        correct_pred[label] += 1

                    total_pred[label] += 1

    accuracy = correct / total

    avg_loss = None

    if criterion is not None:
        avg_loss = running_loss / len(data_loader)

    if final:
        for class_index in range(10):
            class_total = total_pred[class_index]

            if class_total > 0:
                class_accuracy = 100 * correct_pred[class_index] / class_total
            else:
                class_accuracy = 0.0

            print(f"Accuracy class {class_index} = " f"{class_accuracy:.1f} %")

    return accuracy, avg_loss
