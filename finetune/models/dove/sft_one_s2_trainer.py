from .lora_one_s2_trainer import DOVES2Trainer
from ..utils import register

class DOVES2SFTTrainer(DOVES2Trainer):
    pass

register("dove-s2", "sft", DOVES2SFTTrainer)
