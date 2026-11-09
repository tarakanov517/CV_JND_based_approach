"""Shared, explicit run manifest; paths relative to the experiment output root."""
SIGMAS = (0.25, 0.50, 0.85, 1.0, 2.5)
SEEDS = (42, 43, 44)
ORDERS = {
    "forward": ("V1", "V2", "V4", "IT"),
    "reverse": ("IT", "V4", "V2", "V1"),
    "v2_first": ("V2", "V1", "V4", "IT"),
}


def training_jobs():
    jobs = []
    for seed in SEEDS:
        jobs.append(dict(kind="conditional", name="conditional_axon", seed=seed))
        for sigma in SIGMAS:
            for kind in ("fixed", "film_fixed"):
                jobs.append(dict(kind=kind, name=f"{kind}_{int(round(sigma * 100)):03d}",
                                 seed=seed, profile=[sigma] * 4))
        for order in ORDERS:
            jobs.append(dict(kind="adaptive", name=f"adaptive_{order}", seed=seed,
                             order=ORDERS[order]))
    return jobs


def evaluation_jobs():
    jobs = []
    for training in training_jobs():
        profiles = ([(f"all_{int(round(s * 100)):03d}", [s] * 4) for s in SIGMAS]
                    + [("zero", [0.0] * 4)] if training["kind"] == "conditional"
                    else [("trained", None)])
        for label, profile in profiles:
            for mode in ("off", "on"):
                jobs.append(dict(**training, eval_profile=profile,
                                 profile_label=label, noise_mode=mode))
        if training["kind"] == "conditional":
            for order in ORDERS:
                for mode in ("off", "on"):
                    jobs.append(dict(**training, eval_profile=None,
                                     profile_label=f"matched_adaptive_{order}",
                                     profile_from=f"adaptive_{order}/seed_{training['seed']}/result.json",
                                     noise_mode=mode))
    # One epsilon per task keeps strong EOT attacks within individual time limits.
    return [dict(job, epsilon=epsilon) for job in jobs for epsilon in (2, 4, 8)]
