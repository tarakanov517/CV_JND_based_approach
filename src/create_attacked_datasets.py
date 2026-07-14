from attacks import fgsm_attack, pgd_attack

def create_adversarial_dataset(
    attack_name,
    model,
    data_loader,
    criterion,
    device,
    epsilon,
    alpha=None,
    num_iter=None,
):
    adversarial_data = []

    attack_name = attack_name.lower()

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if attack_name == "fgsm":
            adversarial_inputs, labels = fgsm_attack(
                model=model,
                inputs=inputs,
                labels=labels,
                criterion=criterion,
                epsilon=epsilon,
            )

        elif attack_name == "pgd":
            if alpha is None:
                raise ValueError(
                    "Для PGD нужно передать alpha"
                )

            if num_iter is None:
                raise ValueError(
                    "Для PGD нужно передать num_iter"
                )

            adversarial_inputs, labels = pgd_attack(
                model=model,
                inputs=inputs,
                labels=labels,
                criterion=criterion,
                epsilon=epsilon,
                alpha=alpha,
                num_iter=num_iter,
                device=device,
            )

        else:
            raise ValueError(
                f"Неизвестная атака: {attack_name}"
            )

        adversarial_data.append(
            (
                adversarial_inputs.detach().cpu(),
                labels.detach().cpu(),
            )
        )

    return adversarial_data