import super_gradients
from super_gradients import init_trainer, Trainer
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training import models, dataloaders
from super_gradients.common.data_types.enum import MultiGPUMode, StrictLoad
import numpy as np


def main():
    init_trainer()
    setup_device(device=None, multi_gpu="DDP", num_gpus=8)

    trainer = Trainer(experiment_name="coco2017_yolo_nas_s", ckpt_root_dir="None")

    num_classes = 80
    arch_params = {
        "in_channels": 3,
        "backbone": {
            "NStageBackbone": {
                "stem": {"YoloNASStem": {"out_channels": 48}},
                "stages": [
                    {"YoloNASStage": {"out_channels": 96, "num_blocks": 2, "activation_type": "relu", "hidden_channels": 32, "concat_intermediates": False}},
                    {"YoloNASStage": {"out_channels": 192, "num_blocks": 3, "activation_type": "relu", "hidden_channels": 64, "concat_intermediates": False}},
                    {"YoloNASStage": {"out_channels": 384, "num_blocks": 5, "activation_type": "relu", "hidden_channels": 96, "concat_intermediates": False}},
                    {"YoloNASStage": {"out_channels": 768, "num_blocks": 2, "activation_type": "relu", "hidden_channels": 192, "concat_intermediates": False}},
                ],
                "context_module": {"SPP": {"output_channels": 768, "activation_type": "relu", "k": [5, 9, 13]}},
                "out_layers": ["stage1", "stage2", "stage3", "context_module"],
            }
        },
        "neck": {
            "YoloNASPANNeckWithC2": {
                "neck1": {
                    "YoloNASUpStage": {
                        "out_channels": 192,
                        "num_blocks": 2,
                        "hidden_channels": 64,
                        "width_mult": 1,
                        "depth_mult": 1,
                        "activation_type": "relu",
                        "reduce_channels": True,
                    }
                },
                "neck2": {
                    "YoloNASUpStage": {
                        "out_channels": 96,
                        "num_blocks": 2,
                        "hidden_channels": 48,
                        "width_mult": 1,
                        "depth_mult": 1,
                        "activation_type": "relu",
                        "reduce_channels": True,
                    }
                },
                "neck3": {
                    "YoloNASDownStage": {
                        "out_channels": 192,
                        "num_blocks": 2,
                        "hidden_channels": 64,
                        "activation_type": "relu",
                        "width_mult": 1,
                        "depth_mult": 1,
                    }
                },
                "neck4": {
                    "YoloNASDownStage": {
                        "out_channels": 384,
                        "num_blocks": 2,
                        "hidden_channels": 64,
                        "activation_type": "relu",
                        "width_mult": 1,
                        "depth_mult": 1,
                    }
                },
            }
        },
        "heads": {
            "NDFLHeads": {
                "num_classes": 80,
                "reg_max": 16,
                "heads_list": [
                    {"YoloNASDFLHead": {"inter_channels": 128, "width_mult": 0.5, "first_conv_group_size": 0, "stride": 8}},
                    {"YoloNASDFLHead": {"inter_channels": 256, "width_mult": 0.5, "first_conv_group_size": 0, "stride": 16}},
                    {"YoloNASDFLHead": {"inter_channels": 512, "width_mult": 0.5, "first_conv_group_size": 0, "stride": 32}},
                ],
            }
        },
        "bn_eps": 0.001,
        "bn_momentum": 0.03,
        "inplace_act": True,
        "_convert_": "all",
        "num_classes": 80,
    }

    model = models.get(
        model_name="yolo_nas_s",
        num_classes=num_classes,
        arch_params=arch_params,
        strict_load=StrictLoad.NO_KEY_MATCHING,
        pretrained_weights=None,
        checkpoint_path=None,
        load_backbone=False,
        checkpoint_num_classes=None,
    )

    train_dataloader = dataloaders.get(
        name="coco2017_train_yolo_nas",
        dataset_params={
            "data_dir": "/data/coco",
            "subdir": "images/train2017",
            "json_file": "instances_train2017.json",
            "input_dim": [640, 640],
            "cache_dir": None,
            "cache": False,
            "cache_annotations": True,
            "ignore_empty_annotations": True,
            "transforms": [
                {
                    "DetectionRandomAffine": {
                        "degrees": 0,
                        "translate": 0.25,
                        "scales": [0.5, 1.5],
                        "shear": 0.0,
                        "target_size": None,
                        "filter_box_candidates": True,
                        "wh_thr": 2,
                        "area_thr": 0.1,
                        "ar_thr": 20,
                    }
                },
                {"DetectionRGB2BGR": {"prob": 0.5}},
                {"DetectionHSV": {"prob": 0.5, "hgain": 18, "sgain": 30, "vgain": 30}},
                {"DetectionHorizontalFlip": {"prob": 0.5}},
                {"DetectionMixup": {"input_dim": None, "mixup_scale": [0.5, 1.5], "prob": 0.5, "flip_prob": 0.5}},
                {"DetectionPaddedRescale": {"input_dim": [640, 640], "pad_value": 114}},
                {"DetectionStandardize": {"max_value": 255.0}},
                {"DetectionTargetsFormatTransform": {"output_format": "LABEL_CXCYWH"}},
            ],
            "tight_box_rotation": False,
            "class_inclusion_list": None,
            "max_num_samples": None,
            "with_crowd": False,
        },
        dataloader_params={"batch_size": 32, "num_workers": 8, "shuffle": True, "drop_last": True, "pin_memory": True, "collate_fn": "DetectionCollateFN"},
    )

    val_dataloader = dataloaders.get(
        name="coco2017_val_yolo_nas",
        dataset_params={
            "data_dir": "/data/coco",
            "subdir": "images/val2017",
            "json_file": "instances_val2017.json",
            "input_dim": [636, 636],
            "cache_dir": None,
            "cache": False,
            "cache_annotations": True,
            "ignore_empty_annotations": True,
            "transforms": [
                {"DetectionRGB2BGR": {"prob": 1}},
                {"DetectionPadToSize": {"output_size": [640, 640], "pad_value": 114}},
                {"DetectionStandardize": {"max_value": 255.0}},
                "DetectionImagePermute",
                {"DetectionTargetsFormatTransform": {"input_dim": [640, 640], "output_format": "LABEL_CXCYWH"}},
            ],
            "tight_box_rotation": False,
            "class_inclusion_list": None,
            "max_num_samples": None,
            "with_crowd": True,
        },
        dataloader_params={
            "batch_size": 25,
            "num_workers": 8,
            "drop_last": False,
            "shuffle": False,
            "pin_memory": True,
            "collate_fn": "CrowdDetectionCollateFN",
        },
    )

    _lr_updates = super_gradients.training.utils.utils.empty_list()

    _valid_metrics_list_0_detectionmetrics_post_prediction_callback = super_gradients.training.models.detection_models.pp_yolo_e.PPYoloEPostPredictionCallback(
        score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7
    )

    training_hyperparams = {
        "resume": None,
        "run_id": None,
        "resume_path": None,
        "resume_from_remote_sg_logger": False,
        "ckpt_name": "ckpt_latest.pth",
        "lr_mode": "CosineLRScheduler",
        "lr_schedule_function": None,
        "lr_warmup_epochs": 0,
        "lr_warmup_steps": 1000,
        "lr_cooldown_epochs": 0,
        "warmup_initial_lr": 1e-06,
        "step_lr_update_freq": None,
        "cosine_final_lr_ratio": 0.1,
        "warmup_mode": "LinearBatchLRWarmup",
        "lr_updates": _lr_updates,
        "pre_prediction_callback": None,
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 1e-05},
        "load_opt_params": True,
        "zero_weight_decay_on_bias_and_bn": True,
        "loss": "PPYoloELoss",
        "criterion_params": {"use_static_assigner": False, "num_classes": 80, "reg_max": 16},
        "ema": True,
        "ema_params": {"decay": 0.9997, "decay_type": "threshold", "beta": 15},
        "train_metrics_list": [],
        "valid_metrics_list": [
            {
                "DetectionMetrics": {
                    "score_thres": 0.1,
                    "top_k_predictions": 300,
                    "num_cls": 80,
                    "normalize_targets": True,
                    "post_prediction_callback": _valid_metrics_list_0_detectionmetrics_post_prediction_callback,
                }
            }
        ],
        "metric_to_watch": "mAP@0.50:0.95",
        "greater_metric_to_watch_is_better": True,
        "launch_tensorboard": False,
        "tensorboard_port": None,
        "tb_files_user_prompt": False,
        "save_tensorboard_to_s3": False,
        "precise_bn": False,
        "precise_bn_batch_size": None,
        "sync_bn": True,
        "silent_mode": False,
        "mixed_precision": True,
        "save_ckpt_epoch_list": [100, 200, 250],
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
        "max_epochs": 300,
        "initial_lr": 0.0002,
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
