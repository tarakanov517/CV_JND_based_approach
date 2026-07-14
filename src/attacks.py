import torch


def fgsm_attack(model, inputs, labels, criterion, epsilon):
    model.eval()

    inputs = inputs.clone().detach()
    inputs.requires_grad_(True)

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = inputs.grad
    adversarial_inputs = inputs + epsilon * data_grad.sign()

    adversarial_inputs = torch.clamp(adversarial_inputs, -1, 1)

    return adversarial_inputs.detach(), labels


def pgd_attack(
    model,
    inputs,
    labels,
    criterion,
    epsilon,
    device,
    alpha=0.01,
    num_iter=10,
):
    model.eval()

    original_inputs = inputs.clone().detach().to(device)
    labels = labels.to(device)

    adversarial_inputs = original_inputs.clone().detach()

    for _ in range(num_iter):
        adversarial_inputs.requires_grad_(True)

        outputs = model(adversarial_inputs)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adversarial_inputs = (
                adversarial_inputs + alpha * adversarial_inputs.grad.sign()
            )

            perturbation = adversarial_inputs - original_inputs
            perturbation = torch.clamp(
                perturbation,
                min=-epsilon,
                max=epsilon,
            )

            adversarial_inputs = original_inputs + perturbation
            adversarial_inputs = torch.clamp(
                adversarial_inputs,
                min=-1,
                max=1,
            )

        adversarial_inputs = adversarial_inputs.detach()

    return adversarial_inputs, labels
