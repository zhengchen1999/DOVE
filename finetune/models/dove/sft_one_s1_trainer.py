from .lora_one_s1_trainer import DOVES1Trainer
from ..utils import register

class DOVES1SFTTrainer(DOVES1Trainer):
    pass

register("dove-s1", "sft", DOVES1SFTTrainer)
