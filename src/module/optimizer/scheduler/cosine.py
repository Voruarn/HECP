from omegaconf import DictConfig
from torch.optim import Optimizer
from transformers import get_cosine_schedule_with_warmup

from src.module.optimizer.scheduler import register_scheduler


@register_scheduler("cosine")
def cosine_scheduler(optimizer: Optimizer, scheduler_params: DictConfig, max_steps: float, **kwargs) -> dict:
    num_training_steps = int(max_steps)
    num_warmup_steps = scheduler_params.num_warmup_steps if scheduler_params.num_warmup_steps else \
        num_training_steps * scheduler_params.warmup_step_ratio

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        num_cycles=scheduler_params.num_cycles,
    )

    return {
        "interval": "step",
        "scheduler": scheduler,
    }
