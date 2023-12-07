import keras


def init_cosine_decay_lr(
    total_steps: int,
    warmup_steps: int = 10,
    decay_steps_prnt: float = 0.8,
    alpha: float = 0.1,
    warmup_target: float = 1e-2,
):
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-6,
        decay_steps=(total_steps - warmup_steps) * decay_steps_prnt,
        alpha=alpha,
        warmup_steps=warmup_steps,
        warmup_target=warmup_target,
    )
