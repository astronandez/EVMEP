import super_gradients
from super_gradients import init_trainer, Trainer
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training import models, dataloaders
from super_gradients.common.data_types.enum import MultiGPUMode, StrictLoad
import numpy as np


def main():
    init_trainer()
    setup_device(device=None, multi_gpu="DDP", num_gpus=2)

    trainer = Trainer(experiment_name="mobileNetv3_small_training", ckpt_root_dir="None")

    num_classes = 1000
    arch_params = {
        "structure": [
            [3, 1, 16, 1, 0, 2],
            [3, 4.5, 24, 0, 0, 2],
            [3, 3.67, 24, 0, 0, 1],
            [5, 4, 40, 1, 1, 2],
            [5, 6, 40, 1, 1, 1],
            [5, 6, 40, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 6, 96, 1, 1, 2],
            [5, 6, 96, 1, 1, 1],
            [5, 6, 96, 1, 1, 1],
        ],
        "mode": "small",
        "num_classes": 1000,
        "width_mult": 1,
        "dropout": 0.2,
    }

    model = models.get(
        model_name="mobilenet_v3_small",
        num_classes=num_classes,
        arch_params=arch_params,
        strict_load=StrictLoad.NO_KEY_MATCHING,
        pretrained_weights=None,
        checkpoint_path=None,
        load_backbone=False,
        checkpoint_num_classes=None,
    )

    train_dataloader = dataloaders.get(
        name="imagenet_train",
        dataset_params={
            "root": "/data/Imagenet/train",
            "transforms": [
                {"RandomResizedCropAndInterpolation": {"size": 224, "interpolation": "default"}},
                "RandomHorizontalFlip",
                "ToTensor",
                {"Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
            ],
        },
        dataloader_params={"shuffle": True, "batch_size": 128, "num_workers": 16, "drop_last": False, "pin_memory": True},
    )

    val_dataloader = dataloaders.get(
        name="imagenet_val",
        dataset_params={
            "root": "/data/Imagenet/val",
            "transforms": [
                {"Resize": {"size": 256}},
                {"CenterCrop": {"size": 224}},
                "ToTensor",
                {"Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
            ],
        },
        dataloader_params={"batch_size": 200, "num_workers": 16, "drop_last": False, "pin_memory": True},
    )

    _lr_updates = super_gradients.training.utils.utils.empty_list()

    training_hyperparams = {
        "resume": None,
        "run_id": None,
        "resume_path": None,
        "resume_from_remote_sg_logger": False,
        "ckpt_name": "ckpt_latest.pth",
        "lr_mode": "CosineLRScheduler",
        "lr_schedule_function": None,
        "lr_warmup_epochs": 5,
        "lr_warmup_steps": 0,
        "lr_cooldown_epochs": 0,
        "warmup_initial_lr": None,
        "step_lr_update_freq": None,
        "cosine_final_lr_ratio": 0.01,
        "warmup_mode": "LinearEpochLRWarmup",
        "lr_updates": _lr_updates,
        "pre_prediction_callback": None,
        "optimizer": "SGD",
        "optimizer_params": {"weight_decay": 4e-05},
        "load_opt_params": True,
        "zero_weight_decay_on_bias_and_bn": True,
        "loss": "LabelSmoothingCrossEntropyLoss",
        "criterion_params": {"smooth_eps": 0.1},
        "ema": True,
        "ema_params": {"decay": 0.9999, "decay_type": "exp", "beta": 15},
        "train_metrics_list": ["Accuracy", "Top5"],
        "valid_metrics_list": ["Accuracy", "Top5"],
        "metric_to_watch": "Accuracy",
        "greater_metric_to_watch_is_better": True,
        "launch_tensorboard": False,
        "tensorboard_port": None,
        "tb_files_user_prompt": False,
        "save_tensorboard_to_s3": False,
        "precise_bn": False,
        "precise_bn_batch_size": None,
        "sync_bn": False,
        "silent_mode": False,
        "mixed_precision": False,
        "save_ckpt_epoch_list": [],
        "average_best_models": True,
        "dataset_statistics": False,
        "batch_accumulate": 1,
        "run_validation_freq": 1,
        "run_test_freq": 1,
        "save_model": True,
        "seed": 42,
        "phase_callbacks": [],
        "log_installed_packages": True,
        "clip_grad_norm": None,
        "ckpt_best_name": "ckpt_best.pth",
        "max_train_batches": None,
        "max_valid_batches": None,
        "sg_logger": "base_sg_logger",
        "sg_logger_params": {
            "tb_files_user_prompt": False,
            "launch_tensorboard": False,
            "tensorboard_port": None,
            "save_checkpoints_remote": False,
            "save_tensorboard_remote": False,
            "save_logs_remote": False,
            "monitor_system": True,
        },
        "torch_compile": False,
        "torch_compile_loss": False,
        "torch_compile_options": {"mode": "reduce-overhead", "fullgraph": False, "dynamic": False, "backend": "inductor", "options": None, "disable": False},
        "_convert_": "all",
        "max_epochs": 150,
        "initial_lr": 0.1,
    }

    # TRAIN
    result = trainer.train(
        model=model,
        train_loader=train_dataloader,
        valid_loader=val_dataloader,
        training_params=training_hyperparams,
    )

    print(result)


if __name__ == "__main__":
    main()
