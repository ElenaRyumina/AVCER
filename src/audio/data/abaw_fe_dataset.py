import sys

sys.path.append("src/audio")

import os
import numpy as np
import pandas as pd

import cv2
import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from config import c_config
from utils.common_utils import round_math, array_to_bytes, bytes_to_array


class AbawFEDataset(Dataset):
    """Dataset for feature extraction
    Preprocesses labels and features during initialization

    Args:
        audio_root (str): Wavs root dir
        video_root (str): Videos root dir
        labels_root (str | None): Labels root dir
        label_filenames (str): Filenames of labels
        dataset (str): Dataset type. Can be 'Train' or 'Validation'
        features_root (str): Features root dir
        sr (int, optional): Sample rate of audio files. Defaults to 16000.
        shift (int, optional): Window shift in seconds. Defaults to 4.
        min_w_len (int, optional): Minimum window length in seconds. Defaults to 2.
        max_w_len (int, optional): Maximum window length in seconds. Defaults to 4.
        num_classes (int, optional): Maximum number of classess. Defaults to 8.
        transform (torchvision.transforms.transforms.Compose, optional): transform object. Defaults to None.
        processor_name (str, optional): Name of model in transformers library. Defaults to 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'.
    """

    def __init__(
        self,
        audio_root: str,
        video_root: str,
        labels_root: str | None,
        label_filenames: str,
        dataset: str,
        features_root: str,
        sr: int = 16000,
        shift: int = 4,
        min_w_len: int = 2,
        max_w_len: int = 4,
        num_classes: int = 8,
        transform: torchvision.transforms.transforms.Compose = None,
        processor_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    ) -> None:
        self.audio_root = audio_root
        self.video_root = video_root
        self.labels_root = labels_root
        self.label_filenames = label_filenames
        self.dataset = dataset
        self.features_root = features_root

        self.sr = sr
        self.shift = shift
        self.min_w_len = min_w_len
        self.max_w_len = max_w_len

        self.num_classes = num_classes

        self.transform = transform

        self.meta = []
        self.expr_labels = []
        self.new_fps = 5  # downsampling to fps per second
        self.expr_labels_counts = []

        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)

        self.prepare_data()

    def parse_features(
        self, lab_feat_df: pd.core.frame.DataFrame, lab_filename: str, frame_rate: float
    ) -> tuple[list[dict], list[np.ndarray]]:
        """Creates windows with `shift`, `max_w_len`, `min_w_len` in the following steps:
        - Gets FPS and number of frames
        - Pads labels values to `max_w_len` seconds
        - Downsamples to `self.new_fps`. Removes several labels
        - Drops `timings` duplicates:
            f.e. frame_rate = 30, len(seq) = 76, max_w_len = 4 * 30. In this case we
            will have only 3 seconds of VA.
            seg 0: frames 0 - 60 extended to 4 * 30 and converted to 0 - 76
            seg 1: frames 60 - 76 extended to 4 * 30 and converted to 0 - 76

        Args:
            lab_feat_df (pd.core.frame.DataFrame): Features with labels dataframe
            lab_filename (str): Lab filename
            frame_rate (float): Frame rate of video. Defaults to 30.0.

        Returns:
            (list[dict], list[np.ndarray]): Created list of window info (lab_filename, start_t, end_t, start_f, end_f, expr) and expression labels
        """
        shift = self.shift * round_math(frame_rate)
        max_w_len = self.max_w_len * round_math(frame_rate)
        min_w_len = self.min_w_len * round_math(frame_rate)

        timings = []

        frames = (
            lab_feat_df["lab_id"].astype(int).to_list()
        )  # lab_id is the same as frame
        mouth_open_f = (
            lab_feat_df["mouth_open"].astype(int).values
        )  # lab_id is the same as frame

        downsampled_frames = list(
            map(
                round_math,
                np.arange(
                    0,
                    round_math(frame_rate) * self.max_w_len - 1,
                    round_math(frame_rate) / self.new_fps,
                    dtype=float,
                ),
            )
        )

        exprs = lab_feat_df["expr"].values

        for seg in range(0, len(frames), shift):
            expr_window = exprs[seg : seg + max_w_len]
            mouth_open_window = mouth_open_f[seg : seg + max_w_len]

            start = frames[seg]
            end_idx = seg + len(expr_window)
            end = (
                frames[end_idx - 1] if end_idx > len(frames) - 1 else frames[end_idx]
            )  # skip last frame

            if (
                len(expr_window) < max_w_len
            ):  # if less than max_w_len: get last -max_w_len elements
                expr_window = exprs[-max_w_len:]
                mouth_open_window = mouth_open_f[-max_w_len:]

                start = frames[
                    max(0, len(frames) - max_w_len)
                ]  # 0 or frame[-max_w_len]
                end = frames[-1]

            mouth_open = np.pad(
                mouth_open_window,
                (
                    0,
                    max(
                        0,
                        round_math(frame_rate) * self.max_w_len
                        - len(mouth_open_window),
                    ),
                ),
                "edge",
            )
            mouth_open = mouth_open[downsampled_frames]

            mouth_open_w = np.split(
                mouth_open, np.arange(self.new_fps, len(mouth_open), self.new_fps)
            )
            mouth_open = np.asarray(
                [max(set(i), key=list(i).count) for i in mouth_open_w]
            ).T

            exprl = max(set(expr_window), key=list(expr_window).count)
            if exprl > self.num_classes - 1:
                continue

            timings.append(
                {
                    "lab_filename": lab_filename,
                    "fps": frame_rate,
                    "start_t": start / round_math(frame_rate),
                    "end_t": end / round_math(frame_rate),
                    "start_f": start,
                    "end_f": end,
                    "mouth_open": array_to_bytes(mouth_open),
                    "expr": array_to_bytes(np.asarray(exprl)),
                }
            )

        # check duplicates
        # f.e. frame_rate = 30, len(seq) = 76, max_w_len = 4 * 30. In this case we will have only 3 seconds of VA.
        # seg 0: frames 0 - 60 extended to 4 * 30 and converted to 0 - 76
        # seg 1: frames 60 - 76 extended to 4 * 30 and converted to 0 - 76
        timings = [dict(t) for t in {tuple(d.items()) for d in timings}]
        timings = sorted(timings, key=lambda d: d["start_t"])

        expr_labels = []
        for t in timings:
            t["mouth_open"] = bytes_to_array(t["mouth_open"])
            t["expr"] = bytes_to_array(t["expr"])
            expr_labels.append(t["expr"])

        return timings, expr_labels

    def find_corresponding_video_info(self, lab_filename: str) -> tuple[float, float]:
        """Finds video info with corresponding label file in the following steps:
        - Removes extension of file, '_left', '_right' prefixes from label filename
        - Forms list with corresponding video files (with duplicates)
        - Picks first video from video files candidates
        - Gets FPS and total number of frames of video file

        Args:
            lab_filename (str): Label filename

        Returns:
            tuple[float, float]: FPS value and total number of frames
        """
        lab_filename = lab_filename.split(".")[0]
        lab_fns = [lab_filename.split(postfix)[0] for postfix in ["_right", "_left"]]
        res = []
        for l_fn in lab_fns:
            res.extend(
                [v for v in os.listdir(self.video_root) if l_fn == v.split(".")[0]]
            )

        vidcap = cv2.VideoCapture(os.path.join(self.video_root, list(set(res))[0]))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        num_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        return fps, num_frames

    def prepare_data(self) -> None:
        """
        Iterates over labels_root, and prepare data:
        - Reads labels
        - Reads features
        - Merges labels and features
        - Joins obtained data and splits it into windows
        - Forms metadata and labels
        - Forms expr label statistics
        """
        for label_file_name in sorted(self.label_filenames):
            if ".DS_Store" in label_file_name:
                continue

            frame_rate, num_frames = self.find_corresponding_video_info(label_file_name)

            if self.labels_root:
                expr_label_file_path = os.path.join(
                    self.labels_root, self.dataset, label_file_name
                )
                labs = pd.read_csv(
                    expr_label_file_path, sep=".", names=["expr"], header=0
                )
            else:
                labs = pd.DataFrame(
                    data=np.full((int(num_frames), 1), -2), columns=["expr"]
                )

            labs["lab_id"] = labs.index + 1

            features = pd.read_csv(
                os.path.join(self.features_root, label_file_name.replace("txt", "csv")),
                sep=",",
                names=["feat_id", "frame", "surface_area_mouth", "mouth_open"],
                header=0,
            )

            labs_and_feats = labs.merge(
                features, how="left", left_on="lab_id", right_on="frame"
            )
            labs_and_feats[["mouth_open"]] = labs_and_feats[["mouth_open"]].fillna(
                value=0.0
            )

            timings, expr_processed_labs = self.parse_features(
                lab_feat_df=labs_and_feats,
                lab_filename=label_file_name,
                frame_rate=frame_rate,
            )
            self.meta.extend(timings)
            self.expr_labels.extend(expr_processed_labs)

        self.expr_labels_counts = np.unique(
            np.asarray(self.expr_labels), return_counts=True
        )[1]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, list[np.ndarray, np.ndarray], list[dict]]:
        """Gets sample from dataset:
        - Reads audio
        - Selects indexes of audio according to metadata
        - Pads the obtained wav
        - Augments the obtained window
        - Extracts preliminary wav2vec features
        - Drops channel dimension

        Args:
            index (int): Index of sample from metadata

        Returns:
            tuple[torch.FloatTensor, int, list[dict]]: x, Y, sample_info as list for dataloader
        """
        data = self.meta[index]

        wav_path = (
            data["lab_filename"]
            .replace("_right", "")
            .replace("_left", "")
            .replace("txt", "wav")
        )
        a_data, a_data_sr = torchaudio.load(os.path.join(self.audio_root, wav_path))
        a_data = a_data[
            :,
            round(a_data_sr * data["start_t"]) : min(
                round(a_data_sr * data["end_t"]),
                a_data_sr * (data["end_t"] + self.max_w_len),
            ),
        ]  # Due to rounding error fps - cut off window end
        a_data = torch.nn.functional.pad(
            a_data,
            (0, max(0, self.max_w_len * a_data_sr - a_data.shape[1])),
            mode="constant",
        )

        if self.transform:
            a_data = self.transform(a_data)

        wave = self.processor(a_data, sampling_rate=a_data_sr)
        wave = wave["input_values"][0].squeeze()

        sample_info = {
            "filename": os.path.basename(data["lab_filename"]),
            "fps": data["fps"],
            "start_t": data["start_t"],
            "end_t": data["end_t"],
            "start_f": data["start_f"],
            "end_f": data["end_f"],
            "mouth_open": data["mouth_open"],
        }

        y = data["expr"]

        return torch.FloatTensor(wave), y, [sample_info]

    def __len__(self) -> int:
        """Return number of all samples in dataset

        Returns:
            int: Length of meta list
        """
        return len(self.meta)


if __name__ == "__main__":
    ds_names = {"train": "train", "devel": "validation"}

    metadata_info = {}
    for ds in ds_names:
        metadata_info[ds] = {
            "label_filenames": os.listdir(
                os.path.join(
                    c_config["ABAW_LABELS_ROOT"],
                    "{0}_Set".format(ds_names[ds].capitalize()),
                )
            ),
            "dataset": "{0}_Set".format(ds_names[ds].capitalize()),
        }

    # EXPR
    for ds in ds_names:
        fed = AbawFEDataset(
            audio_root=c_config["ABAW_FILTERED_WAV_ROOT"],
            video_root=c_config["ABAW_VIDEO_ROOT"],
            labels_root=c_config["ABAW_LABELS_ROOT"],
            label_filenames=metadata_info[ds]["label_filenames"],
            dataset=metadata_info[ds]["dataset"],
            features_root=c_config["ABAW_FEATURES_ROOT"],
            shift=2,
            min_w_len=2,
            max_w_len=4,
        )

        dl = torch.utils.data.DataLoader(
            fed, batch_size=8, shuffle=False, num_workers=8
        )

        for d in dl:
            pass

        print("{0}, {1}, OK".format(ds, len(fed.meta)))

    ced = AbawFEDataset(
        audio_root=c_config["C_WAV_ROOT"],
        video_root=c_config["C_VIDEO_ROOT"],
        labels_root=c_config["C_LABELS_ROOT"],
        label_filenames=[
            f.replace("mp4", "txt") for f in os.listdir(c_config["C_VIDEO_ROOT"])
        ],
        dataset=None,
        features_root=c_config["C_FEATURES_ROOT"],
        shift=2,
        min_w_len=2,
        max_w_len=4,
    )

    dl = torch.utils.data.DataLoader(ced, batch_size=8, shuffle=False, num_workers=8)

    for d in dl:
        pass

    print("{0}, {1}, OK".format(ds, len(ced.meta)))
