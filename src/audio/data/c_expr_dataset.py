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

from utils.common_utils import round_math


class CExprDataset(Dataset):
    """C dataset
    Preprocesses labels and features during initialization

    Args:
        audio_root (str): Wavs root dir
        video_root (str): Videos root dir
        labels_root (str | None): Labels root dir
        features_root (str): Features root dir
        sr (int, optional): Sample rate of audio files. Defaults to 16000.
        shift (int, optional): Window shift in seconds. Defaults to 4.
        min_w_len (int, optional): Minimum window length in seconds. Defaults to 2.
        max_w_len (int, optional): Maximum window length in seconds. Defaults to 4.
        transform (torchvision.transforms.transforms.Compose, optional): transform object. Defaults to None.
        processor_name (str, optional): Name of model in transformers library. Defaults to 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'.
    """

    def __init__(
        self,
        audio_root: str,
        video_root: str,
        labels_root: str | None,
        features_root: str,
        sr: int = 16000,
        shift: int = 4,
        min_w_len: int = 2,
        max_w_len: int = 4,
        transform: torchvision.transforms.transforms.Compose = None,
        processor_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    ) -> None:
        self.audio_root = audio_root
        self.video_root = video_root
        self.labels_root = labels_root
        self.features_root = features_root

        self.sr = sr
        self.shift = shift
        self.min_w_len = min_w_len
        self.max_w_len = max_w_len

        self.transform = transform

        self.meta = []
        self.expr_labels = []
        self.threshold = 0.5  # num of seconds with open mouth for threshold. 0 - default, without threshold

        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)

        self.prepare_data()

    def parse_features(
        self, lab_feat_df: pd.core.frame.DataFrame, lab_filename: str, frame_rate: float
    ) -> tuple[list[dict], list]:
        """Creates windows with `shift`, `max_w_len`, `min_w_len` in the following steps:
        - Filters frames with mouth_open, and with labels using threshold
        - Splits data on consecutive row values (based on lab_id):
            [0, 1, 2, 6, 10, 11, 12, 14, 15, 16] -> [[0, 1, 2], [6], [10, 11, 12], [14, 15, 16]]
        - Splits obtained sequences with `shift`, `max_w_len`, `min_w_len`:
            skips sequence with length less than `min_w_len`
            or
            splits sequence on windows:
                `seg` - pointer of label index where window is started,
                        iterates from 0 to len of sequence (or labels) with step of `shift`
                `expr_window` - window with indexes from `seg` to `seg + max_w_len`
                `start` - pointer for first frame number of window (frame number + `seg`)
                `end` - pointer for last frame number of window without last element
                        (frame number + `seg` + len(`expr_window`) - 1)

                if length of obtained windowed labels (`expr_window`) less than `min_w_len`
                forms window from the end of sequence:
                    `expr_window` - windows with indexes from end to start with length of `max_w_len`
                    `start` - 0 if `max_w_len` greater than labels length
                              (it means that in sequence there is only one segment with length less than `max_w_len`)
                              len(`frames`) - max_w_len) else
                    `end` - last frame number
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
            tuple[list[dict], list]: Created list of window info (lab_filename, start_t, end_t, start_f, end_f, label), and list of labels
        """
        shift = self.shift * round_math(frame_rate)
        max_w_len = self.max_w_len * round_math(frame_rate)
        min_w_len = self.min_w_len * round_math(frame_rate)

        # filter mouth_open and mislabeled
        mouth_open_threshold = self.threshold * round_math(frame_rate)
        lab_feat_df["mouth_closed"] = 1 - lab_feat_df["mouth_open"]
        s = lab_feat_df["mouth_closed"].diff().ne(0).cumsum()
        lab_feat_df = lab_feat_df[
            (lab_feat_df["expr"] != -1)
            & (
                (s.groupby(s).transform("size") < mouth_open_threshold)
                | (lab_feat_df["mouth_open"] == 1)
            )
        ]

        # Split the data frame based on consecutive row values differences
        sequences = dict(
            tuple(lab_feat_df.groupby(lab_feat_df["lab_id"].diff().gt(1).cumsum()))
        )

        timings = []
        for idx, s in sequences.items():
            frames = s["lab_id"].astype(int).to_list()
            exprs = s["expr"].to_list()

            if len(frames) < min_w_len:  # less than min_w_len
                continue

            for seg in range(0, len(frames), shift):
                expr_window = exprs[seg : seg + max_w_len]
                start = frames[seg]
                end_idx = seg + len(expr_window)
                end = (
                    frames[end_idx - 1]
                    if end_idx > len(frames) - 1
                    else frames[end_idx]
                )  # skip last frame

                if (
                    len(expr_window) < min_w_len
                ):  # if less than min_w_len: get last -max_w_len elements
                    expr_window = exprs[-max_w_len:]
                    start = frames[
                        max(0, len(frames) - max_w_len)
                    ]  # 0 or frame[-max_w_len]
                    end = frames[-1]

                exprl = max(set(expr_window), key=expr_window.count)

                timings.append(
                    {
                        "lab_filename": lab_filename,
                        "start_t": start / round_math(frame_rate),
                        "end_t": end / round_math(frame_rate),
                        "start_f": start,
                        "end_f": end,
                        "expr": exprl,
                    }
                )

        # check duplicates
        # f.e. frame_rate = 30, len(seq) = 76, max_w_len = 4 * 30. In this case we will have only 3 seconds of VA.
        # seg 0: frames 0 - 60 extended to 4 * 30 and converted to 0 - 76
        # seg 1: frames 60 - 76 extended to 4 * 30 and converted to 0 - 76
        timings = [dict(t) for t in {tuple(d.items()) for d in timings}]

        expr_labels = [t["expr"] for t in timings]
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
        Iterates over videos_root, and prepare data:
        - Reads labels OR Creates labels with -2
        - Reads features
        - Merges labels and features
        - Joins obtained data and splits it into windows
        - Forms metadata and labels
        """
        for fp in os.listdir(self.video_root):
            if ".DS_Store" in fp:
                continue

            v_fps, num_frames = self.find_corresponding_video_info(fp)
            if self.labels_root:
                filename = os.path.join(self.labels_root, fp.replace("mp4", "txt"))
                labs = pd.read_csv(filename, sep=".", names=["expr"], header=0)
                labs["lab_id"] = labs.index + 1
            else:
                labs = pd.DataFrame(
                    data=np.full((int(num_frames), 1), -2), columns=["expr"]
                )
                labs["lab_id"] = labs.index + 1

            features = pd.read_csv(
                os.path.join(self.features_root, fp.replace("mp4", "csv")),
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
                lab_filename=fp.replace("mp4", "txt"),
                frame_rate=v_fps,
            )

            self.meta.extend(timings)
            self.expr_labels.extend(expr_processed_labs)

        self.expr_labels_counts = np.unique(
            np.asarray(self.expr_labels), return_counts=True
        )[1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, list[dict]]:
        """Gets sample from dataset:
        - Reads audio
        - Selects indexes of audio according to metadata
        - Pads the obtained values to `max_w_len` seconds
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
            :, round(a_data_sr * data["start_t"]) : round(a_data_sr * data["end_t"])
        ]
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
            "filename": data["lab_filename"],
            "start_t": data["start_t"],
            "end_t": data["end_t"],
            "start_f": data["start_f"],
            "end_f": data["end_f"],
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
    ced = CExprDataset(
        audio_root=c_config["C_WAV_ROOT"],
        video_root=c_config["C_VIDEO_ROOT"],
        labels_root=c_config["C_LABELS_ROOT"],
        features_root=c_config["C_FEATURES_ROOT"],
        shift=2,
        min_w_len=2,
        max_w_len=4,
    )

    dl = torch.utils.data.DataLoader(ced, batch_size=8, shuffle=False, num_workers=8)

    for d in dl:
        pass

    print("OK")
