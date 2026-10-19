import torch
import torch.nn.functional as F


def evaluate(model, loader, device, attack=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if attack is not None:
            inputs = attack(model, inputs, targets)
        with torch.no_grad():
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.size(0)
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def fgsm(epsilon=8 / 255):
    def attack(model, inputs, targets):
        adversarial = inputs.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), targets)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        return (adversarial + epsilon * gradient.sign()).clamp(0, 1).detach()

    return attack


def pgd(epsilon=8 / 255, alpha=2 / 255, steps=10):
    def attack(model, inputs, targets):
        adversarial = inputs + torch.empty_like(inputs).uniform_(-epsilon, epsilon)
        adversarial = adversarial.clamp(0, 1).detach()
        for _ in range(steps):
            adversarial.requires_grad_(True)
            loss = F.cross_entropy(model(adversarial), targets)
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = adversarial.detach() + alpha * gradient.sign()
            delta = (adversarial - inputs).clamp(-epsilon, epsilon)
            adversarial = (inputs + delta).clamp(0, 1).detach()
        return adversarial

    return attack
