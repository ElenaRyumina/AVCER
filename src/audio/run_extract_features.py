import sys

sys.path.append("../src")

import os
import pickle
import numpy as np

from tqdm import tqdm
import torch
import torch.nn.functional as F

from config import c_config, afew_config
from copy import deepcopy

from data.abaw_fe_dataset import AbawFEDataset
from data.afew_fe_dataset import AfewFEDataset

from net_trainer.net_trainer import NetTrainer, ProblemType

from models.audio_expr_models import ExprModelV1 as ExprModelV18
from models.audio_expr_models import ExprModelV2 as ExprModelV28
from models.audio_expr_models import ExprModelV3 as ExprModelV38

from models.audio_expr_models_7_cl import ExprModelV1 as ExprModelV17
from models.audio_expr_models_7_cl import ExprModelV2 as ExprModelV27
from models.audio_expr_models_7_cl import ExprModelV3 as ExprModelV37

from utils.accuracy_utils import recall, precision, f1
from utils.common_utils import define_seed

from transformers import logging

logging.set_verbosity_error()


def main(model_params, c_conf, afew_conf) -> None:
    abaw_audio_root = (
        c_conf["ABAW_FILTERED_WAV_ROOT"]
        if c_conf["FILTERED"]
        else c_conf["ABAW_WAV_ROOT"]
    )
    abaw_video_root = c_conf["ABAW_VIDEO_ROOT"]
    abaw_labels_root = c_conf["ABAW_LABELS_ROOT"]
    abaw_features_root = c_conf["ABAW_FEATURES_ROOT"]

    logs_root = c_conf["LOGS_ROOT"]

    c_audio_root = (
        c_conf["C_FILTERED_WAV_ROOT"] if c_conf["FILTERED"] else c_conf["C_WAV_ROOT"]
    )
    c_video_root = c_conf["C_VIDEO_ROOT"]
    c_labels_root = c_conf["C_LABELS_ROOT"]
    c_features_root = c_conf["C_FEATURES_ROOT"]

    afew_audio_root = (
        afew_conf["AFEW_FILTERED_WAV_ROOT"]
        if afew_conf["FILTERED"]
        else afew_conf["AFEW_WAV_ROOT"]
    )
    afew_vad_root = afew_conf["AFEW_VAD_ROOT"]

    seven_classes = "7cl" in model_params["model_name"]
    model_name = c_conf["MODEL_PARAMS"]["args"]["model_name"]
    batch_size = c_conf["BATCH_SIZE"]

    ds_names = {
        "abaw_train": "train",
        "abaw_devel": "validation",
        "abaw_c": "devel",
        "afew_train": "train",
        "afew_devel": "val",
    }

    metadata_info = {}
    all_transforms = {}
    for ds in ds_names:
        if "afew" in ds:
            metadata_info[ds] = {
                "audio_root": os.path.join(
                    afew_audio_root,
                    "{0}_AFEW_{1}".format(
                        ds_names[ds].capitalize(),
                        "vocals" if afew_conf["FILTERED"] else "wavs",
                    ),
                ),
                "vad_file_path": os.path.join(
                    afew_vad_root,
                    "vad_{0}_AFEW_{1}_16000.pickle".format(
                        ds_names[ds].capitalize(),
                        "vocals" if afew_conf["FILTERED"] else "wavs",
                    ),
                ),
            }
        elif "abaw_c" in ds:
            metadata_info[ds] = {
                "label_filenames": [
                    f.replace("mp4", "txt") for f in os.listdir(c_video_root)
                ],
                "dataset": None,
            }
        else:
            metadata_info[ds] = {
                "label_filenames": os.listdir(
                    os.path.join(
                        abaw_labels_root, "{0}_Set".format(ds_names[ds].capitalize())
                    )
                ),
                "dataset": "{0}_Set".format(ds_names[ds].capitalize()),
            }

        all_transforms[ds] = None

    datasets = {}
    for ds in ds_names:
        if "afew" in ds:
            datasets[ds] = AfewFEDataset(
                audio_root=metadata_info[ds]["audio_root"],
                vad_file_path=metadata_info[ds]["vad_file_path"],
                #  num_classes=7 if seven_classes else 8,
                shift=2,
                min_w_len=2,
                max_w_len=4,
            )
        elif "abaw_c" in ds:
            datasets[ds] = AbawFEDataset(
                audio_root=c_audio_root,
                video_root=c_video_root,
                labels_root=c_labels_root,
                label_filenames=metadata_info[ds]["label_filenames"],
                dataset=metadata_info[ds]["dataset"],
                features_root=c_features_root,
                #  num_classes=7 if seven_classes else 8,
                shift=2,
                min_w_len=2,
                max_w_len=4,
            )
        else:
            datasets[ds] = AbawFEDataset(
                audio_root=abaw_audio_root,
                video_root=abaw_video_root,
                labels_root=abaw_labels_root,
                label_filenames=metadata_info[ds]["label_filenames"],
                dataset=metadata_info[ds]["dataset"],
                features_root=abaw_features_root,
                #  num_classes=7 if seven_classes else 8,
                shift=2,
                min_w_len=2,
                max_w_len=4,
            )

    if not seven_classes:
        c_names = [
            "Neutral",
            "Anger",
            "Disgust",
            "Fear",
            "Happiness",
            "Sadness",
            "Surprise",
        ]
    else:
        c_names = [
            "Neutral",
            "Anger",
            "Disgust",
            "Fear",
            "Happiness",
            "Sadness",
            "Surprise",
            "Other",
        ]

    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.multiprocessing.set_sharing_strategy("file_system")

    net_trainer = NetTrainer(
        log_root=logs_root,
        experiment_name="Test",
        problem_type=ProblemType.CLASSIFICATION,
        c_names=c_names,
        metrics=None,
        device=device,
        group_predicts_fn=None,
        source_code=None,
    )

    dataloaders = {}
    for ds, v in datasets.items():
        dataloaders[ds] = torch.utils.data.DataLoader(
            v,
            batch_size=batch_size,
            shuffle=("train" in ds),
            num_workers=batch_size if batch_size < 9 else 8,
        )

    model = model_params["model_cls"].from_pretrained(model_name)
    model.load_state_dict(
        torch.load(
            os.path.join(
                model_params["root_path"], "epoch_{}.pth".format(model_params["epoch"])
            )
        )["model_state_dict"]
    )

    model.to(device)

    net_trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    net_trainer.model = model

    keys_mapping = {
        "fps": "fps",
        "vad_info": "vad_info",
        "start_f": "frame_start",
        "end_f": "frame_end",
        "start_t": "timestep_start",
        "end_t": "timestep_end",
        "mouth_open": "mouth_open",
    }

    for ds, v in dataloaders.items():
        targets, predicts, features, sample_info = net_trainer.extract_features(
            phase="test", dataloader=v, verbose=True
        )

        new_sample_info = {}
        for s_idx, si in enumerate(sample_info):
            for idx, fn in enumerate(si["filename"]):
                if fn not in new_sample_info:
                    new_sample_info[fn] = {}
                    new_sample_info[fn]["targets"] = []
                    new_sample_info[fn]["predicts"] = []
                    new_sample_info[fn]["features"] = []
                    for k in si.keys():
                        if "filename" in k:
                            continue

                        new_sample_info[fn][keys_mapping[k]] = []

                new_sample_info[fn]["targets"].append(targets[idx + s_idx * batch_size])
                new_sample_info[fn]["predicts"].append(
                    predicts[idx + s_idx * batch_size]
                )
                new_sample_info[fn]["features"].append(
                    features[idx + s_idx * batch_size]
                )
                for k in si.keys():
                    if "filename" in k:
                        continue

                    if k in ["start_f", "end_f"]:
                        new_sample_info[fn][keys_mapping[k]].append(int(si[k][idx]))
                    elif k in ["fps", "start_t", "end_t"]:
                        new_sample_info[fn][keys_mapping[k]].append(float(si[k][idx]))
                    elif k in ["mouth_open"]:
                        new_sample_info[fn][keys_mapping[k]].append(si[k][idx].numpy())
                    else:
                        new_sample_info[fn][keys_mapping[k]].append(si[k][idx])

        with open(
            os.path.join(
                logs_root,
                "{0}{1}_{2}_as8.pickle".format(
                    model_params["model_name"], "_F" if c_conf["FILTERED"] else "", ds
                ),
            ),
            "wb",
        ) as handle:
            pickle.dump(new_sample_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parameters = [
        # {'model_name': 'FLW-ExprModelV3-2024.03.02-11.42.11', 'model_cls': ExprModelV38, 'epoch': 63, 'filtered': False},
        # {'model_name': 'FLW-ExprModelV3-2024.03.01-20.26.51', 'model_cls': ExprModelV38, 'epoch': 26, 'filtered': True},
        # {'model_name': 'CELSW-ExprModelV3-2024.02.28-10.33.12', 'model_cls': ExprModelV38, 'epoch': 85, 'filtered': True},
        # {'model_name': 'CELSWa-ExprModelV3-2024.02.27-20.52.14', 'model_cls': ExprModelV38, 'epoch': 93, 'filtered': False},
        {
            "model_name": "7cl-FLW-ExprModelV2-2024.03.04-11.52.11",
            "model_cls": ExprModelV27,
            "epoch": 51,
            "filtered": False,
        },
        {
            "model_name": "7cl-FLWa-ExprModelV3-2024.03.03-16.39.11",
            "model_cls": ExprModelV37,
            "epoch": 42,
            "filtered": False,
        },
        {
            "model_name": "7cl-CELSW-ExprModelV1-2024.03.06-09.43.01",
            "model_cls": ExprModelV17,
            "epoch": 96,
            "filtered": False,
        },
        {
            "model_name": "7cl-CELSWa-ExprModelV3-2024.03.05-18.41.30",
            "model_cls": ExprModelV37,
            "epoch": 87,
            "filtered": False,
        },
    ]

    for model_params in parameters:
        model_params["root_path"] = os.path.join(
            "/", model_params["model_name"], "models"
        )  # TODO
        print(model_params["model_name"])

        c_cfg = deepcopy(c_config)
        c_cfg["FILTERED"] = model_params["filtered"]

        afew_cfg = deepcopy(afew_config)
        afew_cfg["FILTERED"] = model_params["filtered"]
        main(model_params, c_conf=c_cfg, afew_conf=afew_cfg)
        print()
