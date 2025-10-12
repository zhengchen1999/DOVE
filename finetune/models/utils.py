from typing import Dict, Literal

from finetune.trainer import Trainer


SUPPORTED_MODELS: Dict[str, Dict[str, Trainer]] = {}


def register(model_name: str, training_type: Literal["lora", "sft"], trainer_cls: Trainer):
    """Register a model and its associated functions for a specific training type.

    Args:
        model_name (str): Name of the model to register (e.g. "cogvideox-5b")
        training_type (Literal["lora", "sft"]): Type of training - either "lora" or "sft"
        trainer_cls (Trainer): Trainer class to register.
    """

    # Check if model_name and training_type exists in SUPPORTED_MODELS
    if model_name not in SUPPORTED_MODELS:
        SUPPORTED_MODELS[model_name] = {}
    else:
        if training_type in SUPPORTED_MODELS[model_name]:
            raise ValueError(f"Training type {training_type} already exists for model {model_name}")

    SUPPORTED_MODELS[model_name][training_type] = trainer_cls


def show_supported_models():
    """Print all currently supported models and their training types."""

    print("\nSupported Models:")
    print("================")

    for model_name, training_types in SUPPORTED_MODELS.items():
        print(f"\n{model_name}")
        print("-" * len(model_name))
        for training_type in training_types:
            print(f"  • {training_type}")


def get_model_cls(model_type: str, training_type: Literal["lora", "sft"]) -> Trainer:
    """Get the trainer class for a specific model and training type."""
    if model_type not in SUPPORTED_MODELS:
        print(f"\nModel '{model_type}' is not supported.")
        print("\nSupported models are:")
        for supported_model in SUPPORTED_MODELS:
            print(f"  • {supported_model}")
        raise ValueError(f"Model '{model_type}' is not supported")

    if training_type not in SUPPORTED_MODELS[model_type]:
        print(f"\nTraining type '{training_type}' is not supported for model '{model_type}'.")
        print(f"\nSupported training types for '{model_type}' are:")
        for supported_type in SUPPORTED_MODELS[model_type]:
            print(f"  • {supported_type}")
        raise ValueError(
            f"Training type '{training_type}' is not supported for model '{model_type}'"
        )
    # from pprint import pprint
    # pprint(SUPPORTED_MODELS)
    # # {'cogvideox-i2v': {'lora': <class 'finetune.models.cogvideox_i2v.lora_trainer.CogVideoXI2VLoraTrainer'>,
    # #                'sft': <class 'finetune.models.cogvideox_i2v.sft_trainer.CogVideoXI2VSftTrainer'>},
    # # 'cogvideox-t2v': {'lora': <class 'finetune.models.cogvideox_t2v.lora_trainer.CogVideoXT2VLoraTrainer'>,
    # #                 'sft': <class 'finetune.models.cogvideox_t2v.sft_trainer.CogVideoXT2VSftTrainer'>},
    # # 'cogvideox1.5-i2v': {'lora': <class 'finetune.models.cogvideox1_5_i2v.lora_trainer.CogVideoX1_5I2VLoraTrainer'>,
    # #                     'sft': <class 'finetune.models.cogvideox1_5_i2v.sft_trainer.CogVideoX1_5I2VSftTrainer'>},
    # # 'cogvideox1.5-t2v': {'lora': <class 'finetune.models.cogvideox1_5_t2v.lora_trainer.CogVideoX1_5T2VLoraTrainer'>,
    # #                     'sft': <class 'finetune.models.cogvideox1_5_t2v.sft_trainer.CogVideoX1_5T2VSftTrainer'>}}
    return SUPPORTED_MODELS[model_type][training_type]
