import sys

sys.path.append("../src")

import os
import pprint
import datetime
from copy import deepcopy

import numpy as np

import torch
from torchvision import transforms

from config import c_config

from augmentation.wave_augmentation import (
    RandomChoice,
    PolarityInversion,
    WhiteNoise,
    Gain,
)

from data.abaw_expr_dataset import AbawExprDataset
from data.meld_dataset import MeldDataset

from net_trainer.net_trainer import NetTrainer

from models.audio_expr_models_7_cl import *
from loss.loss import SoftFocalLoss, SoftFocalLossWrapper

from utils.data_utils import get_source_code

from utils.accuracy_utils import recall, precision, f1
from utils.common_utils import define_seed


def main(config: dict) -> None:
    """Trains with configuration in the following steps:
    - Defines datasets names
    - Defines data augmentations
    - Defines ExprDatasets and MeldDatasets
    - Defines NetTrainer
    - Defines Dataloaders
    - Defines model
    - Defines weighted loss, optimizer, scheduler
    - Runs NetTrainer

    Args:
        config (dict): Configuration dictionary
    """
    abaw_audio_root = (
        config["ABAW_FILTERED_WAV_ROOT"]
        if config["FILTERED"]
        else config["ABAW_WAV_ROOT"]
    )
    abaw_video_root = config["ABAW_VIDEO_ROOT"]
    abaw_labels_root = config["ABAW_LABELS_ROOT"]
    abaw_features_root = config["ABAW_FEATURES_ROOT"]

    meld_audio_root = (
        config["MELD_FILTERED_WAV_ROOT"]
        if config["FILTERED"]
        else config["MELD_WAV_ROOT"]
    )
    meld_labels_root = config["MELD_LABELS_ROOT"]
    meld_vad_root = config["MELD_VAD_ROOT"]

    logs_root = config["LOGS_ROOT"]
    model_cls = config["MODEL_PARAMS"]["model_cls"]
    model_name = config["MODEL_PARAMS"]["args"]["model_name"]
    aug = config["AUGMENTATION"]
    num_epochs = config["NUM_EPOCHS"]
    batch_size = config["BATCH_SIZE"]

    c_names = [
        "Neutral",
        "Anger",
        "Disgust",
        "Fear",
        "Happiness",
        "Sadness",
        "Surprise",
    ]

    source_code = "Configuration:\n{0}\n\nSource code:\n{1}".format(
        pprint.pformat(config),
        get_source_code([main, model_cls, AbawExprDataset, NetTrainer]),
    )

    ds_names = {
        "train": "train",
        "abaw_devel": "validation",
        "meld_devel": "dev",
        "meld_test": "test",
    }

    metadata_info = {}
    all_transforms = {}
    for ds in ds_names:
        if "train" in ds:
            metadata_info[ds] = {
                "abaw": os.path.join(
                    abaw_labels_root, "{0}_Set".format(ds_names[ds].capitalize())
                ),
                "meld": os.path.join(
                    meld_labels_root, "{0}_sent_emo.csv".format(ds_names[ds])
                ),
            }

            if aug:
                all_transforms[ds] = [
                    transforms.Compose(
                        [
                            RandomChoice([PolarityInversion(), WhiteNoise(), Gain()]),
                        ]
                    ),
                ]
            else:
                all_transforms[ds] = [None]
        else:
            if "abaw" in ds:
                metadata_info[ds] = os.path.join(
                    abaw_labels_root, "{0}_Set".format(ds_names[ds].capitalize())
                )

            if "meld" in ds:
                metadata_info[ds] = os.path.join(
                    meld_labels_root, "{0}_sent_emo.csv".format(ds_names[ds])
                )

            all_transforms[ds] = None

    datasets = {}
    for ds in ds_names:
        if "train" in ds:
            abaw_datasets = [
                AbawExprDataset(
                    audio_root=abaw_audio_root,
                    video_root=abaw_video_root,
                    labels_root=metadata_info[ds]["abaw"],
                    features_root=abaw_features_root,
                    shift=2,
                    min_w_len=2,
                    max_w_len=4,
                    num_classes=7,
                    processor_name=model_name,
                    transform=t,
                )
                for t in all_transforms[ds]
            ]

            meld_datasets = [
                MeldDataset(
                    audio_root=os.path.join(meld_audio_root, ds_names[ds]),
                    labels_file_path=metadata_info[ds]["meld"],
                    vad_file_path=os.path.join(
                        config["MELD_VAD_ROOT"],
                        "vad_{0}_{1}_16000.pickle".format(
                            ds_names[ds], "vocals" if config["FILTERED"] else "wavs"
                        ),
                    ),
                    shift=2,
                    min_w_len=2,
                    max_w_len=4,
                    num_classes=7,
                    processor_name=model_name,
                    transform=t,
                )
                for t in all_transforms[ds]
            ]

            datasets[ds] = torch.utils.data.ConcatDataset(abaw_datasets + meld_datasets)
        else:
            if "abaw" in ds:
                datasets[ds] = AbawExprDataset(
                    audio_root=abaw_audio_root,
                    video_root=abaw_video_root,
                    labels_root=metadata_info[ds],
                    features_root=abaw_features_root,
                    shift=2,
                    min_w_len=2,
                    max_w_len=4,
                    num_classes=7,
                    processor_name=model_name,
                    transform=all_transforms[ds],
                )

            if "meld" in ds:
                datasets[ds] = MeldDataset(
                    audio_root=os.path.join(meld_audio_root, ds_names[ds]),
                    labels_file_path=metadata_info[ds],
                    vad_file_path=os.path.join(
                        config["MELD_VAD_ROOT"],
                        "vad_{0}_{1}_16000.pickle".format(
                            ds_names[ds], "vocals" if config["FILTERED"] else "wavs"
                        ),
                    ),
                    shift=2,
                    min_w_len=2,
                    max_w_len=4,
                    num_classes=7,
                    processor_name=model_name,
                    transform=all_transforms[ds],
                )

    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment_name = "7cl-FLW{0}{1}-{2}".format(
        "a-" if aug else "-",
        model_cls.__name__.replace("-", "_").replace("/", "_"),
        datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"),
    )

    net_trainer = NetTrainer(
        log_root=logs_root,
        experiment_name=experiment_name,
        c_names=c_names,
        metrics=[f1, recall, precision],
        device=device,
        group_predicts_fn=None,
        source_code=source_code,
    )

    dataloaders = {}
    for ds in ds_names:
        dataloaders[ds] = torch.utils.data.DataLoader(
            datasets[ds],
            batch_size=batch_size,
            shuffle=("train" in ds),
            num_workers=batch_size if batch_size < 9 else 8,
        )

    model = model_cls.from_pretrained(model_name)

    model.to(device)

    class_sample_count = np.sum(
        [dataset.expr_labels_counts for dataset in datasets["train"].datasets], axis=0
    )

    class_weights = torch.Tensor(max(class_sample_count) / class_sample_count).to(
        device
    )
    # loss = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=.2)
    loss = SoftFocalLossWrapper(
        focal_loss=SoftFocalLoss(alpha=class_weights), num_classes=len(c_names)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=0.001 * 0.1
    )

    model, max_perf = net_trainer.run(
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        dataloaders=dataloaders,
        mixup_alpha=0.3 if aug else None,
    )

    for phase in ds_names:
        if "train" in phase:
            continue

        print()
        print(phase.capitalize())
        print("Epoch: {}, Max performance:".format(max_perf[phase]["epoch"]))
        print([metric for metric in max_perf[phase]["performance"]])
        print(
            [
                max_perf[phase]["performance"][metric]
                for metric in max_perf[phase]["performance"]
            ]
        )
        print()


def run_c_training() -> None:
    """Wrapper for training C challenge"""

    model_cls = [ExprModelV1, ExprModelV2, ExprModelV3]

    for augmentation in [True, False]:
        for filtered in [True, False]:
            for m_cls in model_cls:
                cfg = deepcopy(c_config)
                cfg["FILTERED"] = filtered
                cfg["AUGMENTATION"] = augmentation
                cfg["MODEL_PARAMS"]["model_cls"] = m_cls

                main(cfg)


if __name__ == "__main__":
    run_c_training()
    # main(config)
