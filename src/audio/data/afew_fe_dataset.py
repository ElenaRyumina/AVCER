import sys

sys.path.append("src/audio")

import os
import json
import pickle

import numpy as np
import pandas as pd

import cv2
import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from config import afew_config
from utils.common_utils import round_math, array_to_bytes, bytes_to_array


class AfewFEDataset(Dataset):
    """Dataset for feature extraction
    Preprocesses labels and features during initialization

    Args:
        audio_root (str): Wavs root dir
        vad_file_path (str): Path to vad file
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
        vad_file_path: str,
        sr: int = 16000,
        shift: int = 4,
        min_w_len: int = 2,
        max_w_len: int = 4,
        num_classes: int = 8,
        transform: torchvision.transforms.transforms.Compose = None,
        processor_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    ) -> None:
        self.audio_root = audio_root
        self.vad_file_path = vad_file_path

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

    def afew_to_abaw_labels(self, afew_label: str) -> str:
        """Converts label from afew to abaw

        Args:
            afew_label (str): AFEW label

        Returns:
            str: Abaw label
        """
        afew_labels = {
            "Neutral": 0,
            "Angry": 1,
            "Disgust": 2,
            "Fear": 3,
            "Happy": 4,
            "Sad": 5,
            "Surprise": 6,
        }

        return afew_labels[afew_label]

    def prepare_data(self) -> None:
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
        shift = round(self.shift * self.sr)
        max_w_len = round(self.max_w_len * self.sr)
        min_w_len = round(self.min_w_len * self.sr)

        vad_info = {}
        with open(self.vad_file_path, "rb") as handle:
            vad_info = pickle.load(handle)

        for emo_label in os.listdir(self.audio_root):
            for fn in os.listdir(os.path.join(self.audio_root, emo_label)):
                timings = []
                duration = torchaudio.info(
                    os.path.join(self.audio_root, emo_label, fn)
                ).num_frames

                for seg in range(0, duration, shift):
                    start = seg
                    end = min(duration - 1, seg + max_w_len)

                    if (
                        end - start < min_w_len
                    ):  # if less than max_w_len: get last -max_w_len elements
                        start = max(0, duration - 1 - max_w_len)
                        end = duration - 1

                    exprl = self.afew_to_abaw_labels(emo_label)
                    if exprl > self.num_classes - 1:
                        continue

                    timings.append(
                        {
                            "wav_filename": os.path.join(
                                self.audio_root, emo_label, fn
                            ),
                            "start_t": start / self.sr,
                            "end_t": end / self.sr,
                            "start_f": start,
                            "end_f": end,
                            "expr": array_to_bytes(np.asarray(exprl)),
                        }
                    )

                timings = [dict(t) for t in {tuple(d.items()) for d in timings}]
                timings = sorted(timings, key=lambda d: d["start_t"])

                expr_labels = []
                for t in timings:
                    t["expr"] = bytes_to_array(t["expr"])
                    t["vad_info"] = vad_info[os.path.basename(t["wav_filename"])]
                    expr_labels.append(t["expr"])

                self.meta.extend(timings)

        self.expr_labels_counts = np.unique(
            np.asarray(self.expr_labels), return_counts=True
        )[1]
        if self.num_classes > 7:
            self.expr_labels_counts = np.append(self.expr_labels_counts, 0)

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

        wav_path = data["wav_filename"]
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
            "filename": os.path.basename(data["wav_filename"]),
            "start_t": data["start_t"],
            "end_t": data["end_t"],
            "start_f": data["start_f"],
            "end_f": data["end_f"],
            "vad_info": json.dumps(data["vad_info"]),
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
    ds_names = {"train": "train", "devel": "val"}

    metadata_info = {}
    for ds in ds_names:
        metadata_info[ds] = {
            "audio_root": os.path.join(
                afew_config["AFEW_FILTERED_WAV_ROOT"],
                "{0}_AFEW_{1}".format(
                    ds_names[ds].capitalize(),
                    "vocals" if afew_config["FILTERED"] else "wavs",
                ),
            ),
            "vad_file_path": os.path.join(
                afew_config["AFEW_VAD_ROOT"],
                "vad_{0}_AFEW_{1}_16000.pickle".format(
                    ds_names[ds].capitalize(),
                    "vocals" if afew_config["FILTERED"] else "wavs",
                ),
            ),
        }

    # EXPR
    for ds in ds_names:
        fed = AfewFEDataset(
            audio_root=metadata_info[ds]["audio_root"],
            vad_file_path=metadata_info[ds]["vad_file_path"],
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
