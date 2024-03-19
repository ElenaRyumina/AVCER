import sys

sys.path.append("../src")

import os

import numpy as np

from tqdm import tqdm
import torch
import torch.nn.functional as F

from data.c_expr_dataset import CExprDataset

from models.audio_expr_models import *


def iterate_model(
    model: torch.nn.Module,
    device: torch.device,
    phase: str,
    dataloader: torch.utils.data.dataloader.DataLoader,
    group_predicts_fn: callable = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:

    targets = []
    predicts = []
    sample_info = []

    model.eval()

    # Iterate over data.
    for idx, data in enumerate(tqdm(dataloader, disable=not verbose)):
        inps, labs, s_info = data
        if isinstance(inps, list):
            inps = [d.to(device) for d in inps]
        else:
            inps = inps.to(device)

        labs = labs.to(device)
        # forward and backward
        preds = None
        with torch.set_grad_enabled("train" in phase):
            preds = model(inps)

        targets.extend(labs.cpu().numpy())
        preds = F.softmax(preds, dim=1)

        predicts.extend(preds.cpu().detach().numpy())
        sample_info.extend(s_info)

    if group_predicts_fn:
        targets, predicts, sample_info = group_predicts_fn(
            np.asarray(targets), np.asarray(predicts), sample_info
        )

    return targets, predicts, sample_info


def main(model_params: dict, group_predicts_fn: callable = None) -> None:
    db_root_path = "/"  # TODO
    is_filtered = True
    batch_size = 64

    audio_root = (
        os.path.join(db_root_path, "vocals")
        if is_filtered
        else os.path.join(db_root_path, "wavs")
    )
    video_root = os.path.join(db_root_path, "videos")
    features_root = os.path.join(db_root_path, "features", "open_mouth")
    vad_file_path = os.path.join(db_root_path, "vad_16000.pickle")

    ds_names = {
        "test": "test",
    }

    all_transforms = {}
    for ds in ds_names:
        all_transforms[ds] = None

    datasets = {}
    for ds in ds_names:
        datasets[ds] = CExprDataset(
            audio_root=audio_root,
            video_root=video_root,
            labels_root=None,
            features_root=features_root,
            vad_file_path=vad_file_path,
            shift=2,
            min_w_len=2,
            max_w_len=4,
        )

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

    dataloaders = {}
    for ds, v in datasets.items():
        dataloaders[ds] = torch.utils.data.DataLoader(
            v,
            batch_size=batch_size,
            shuffle=("train" in ds),
            num_workers=batch_size if batch_size < 9 else 8,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model_params["model_cls"].from_pretrained(
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    )
    model.load_state_dict(
        torch.load(
            os.path.join(
                model_params["root_path"], "epoch_{}.pth".format(model_params["epoch"])
            )
        )["model_state_dict"]
    )
    model.to(device)

    res = {}
    for ds, v in dataloaders.items():
        targets, predicts, sample_info = iterate_model(
            model=model,
            device=device,
            phase="test",
            dataloader=v,
            group_predicts_fn=group_predicts_fn,
            verbose=False,
        )

        new_sample_info = {}
        for si in sample_info:
            for k, v in si.items():
                if k not in new_sample_info:
                    new_sample_info[k] = []
                new_sample_info[k].extend(v)

        res[ds] = []
        for idx, t in enumerate(targets):
            res[ds].append(
                {
                    "targets": t,
                    "predicts": predicts[idx],
                    "filename": new_sample_info["filename"][idx],
                    "start_t": float(new_sample_info["start_t"][idx]),
                    "end_t": float(new_sample_info["end_t"][idx]),
                    "start_f": int(new_sample_info["start_f"][idx]),
                    "end_f": int(new_sample_info["end_f"][idx]),
                }
            )


if __name__ == "__main__":
    parameters = [
        {
            "model_name": "CESLWExprModelV2-2024.02.22-10.06.13",
            "model_cls": ExprModelV2,
            "epoch": 74,
        },
        {
            "model_name": "CESLWa-ExprModelV2-2024.02.19-07.59.20",
            "model_cls": ExprModelV2,
            "epoch": 100,
        },
        {
            "model_name": "CESLWExprModelV3-2024.02.22-14.14.01",
            "model_cls": ExprModelV3,
            "epoch": 55,
        },
        {
            "model_name": "CESLWa-ExprModelV3-2024.02.22-00.35.01",
            "model_cls": ExprModelV3,
            "epoch": 77,
        },
    ]

    for model_params in parameters:
        model_params["root_path"] = os.path.join(
            "/", model_params["model_name"], "models"
        )  # TODO
        main(model_params)
