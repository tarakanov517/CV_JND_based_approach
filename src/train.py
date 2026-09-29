import copy

import torch
import torch.nn.functional as F

from attacks import evaluate


def set_backbone_trainable(model, trainable):
    for parameter in model.model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.model.decoder.linear.parameters():
        parameter.requires_grad = True


def set_noise_scale(model, scale):
    for module in model.modules():
        if hasattr(module, "noise_scale"):
            module.noise_scale = scale


def get_noise_scale(schedule, phase, phase_epoch, warmup_epochs, ramp_epochs):
    if schedule == "fixed":
        return 1.0
    if phase == "head" or phase_epoch <= warmup_epochs:
        return 0.0
    return min((phase_epoch - warmup_epochs) / max(ramp_epochs, 1), 1.0)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.size(0)
    return total_loss / total_examples, total_correct / total_examples


def fit(
    model,
    train_loader,
    validation_loader,
    device,
    head_epochs,
    full_epochs,
    head_lr,
    full_lr,
    weight_decay,
    noise_schedule,
    noise_warmup_epochs,
    noise_ramp_epochs,
):
    best_accuracy = -1.0
    best_state = None
    history = []
    phases = [
        ("head", head_epochs, head_lr, False),
        ("full", full_epochs, full_lr, True),
    ]
    epoch_number = 0
    for phase, epochs, learning_rate, backbone_trainable in phases:
        if epochs == 0:
            continue
        set_backbone_trainable(model, backbone_trainable)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        for phase_epoch in range(1, epochs + 1):
            epoch_number += 1
            noise_scale = get_noise_scale(
                noise_schedule,
                phase,
                phase_epoch,
                noise_warmup_epochs,
                noise_ramp_epochs,
            )
            set_noise_scale(model, noise_scale)
            train_loss, train_accuracy = train_epoch(
                model, train_loader, optimizer, device
            )
            validation = evaluate(model, validation_loader, device)
            scheduler.step()
            row = {
                "epoch": epoch_number,
                "phase": phase,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "noise_scale": noise_scale,
            }
            history.append(row)
            print(
                f"epoch={epoch_number} phase={phase} "
                f"noise_scale={noise_scale:.3f} "
                f"train_acc={train_accuracy:.4f} val_acc={validation['accuracy']:.4f}",
                flush=True,
            )
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_accuracy
