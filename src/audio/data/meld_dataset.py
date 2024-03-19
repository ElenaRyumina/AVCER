import sys

sys.path.append("src/audio")

import os
import pickle

import numpy as np
import pandas as pd

import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from config import c_config


class MeldDataset(Dataset):
    """Meld dataset
    Preprocesses labels during initialization

    Args:
        audio_root (str): Wavs root dir
        labels_file_path (str): Labels root dir
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
        labels_file_path: str,
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
        self.labels_file_path = labels_file_path
        self.vad_file_path = vad_file_path

        self.sr = sr
        self.shift = shift
        self.min_w_len = min_w_len
        self.max_w_len = max_w_len

        self.num_classes = num_classes

        self.transform = transform

        self.meta = []
        self.expr_labels = []
        self.expr_labels_counts = []

        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)

        self.prepare_data()

    def meld_to_abaw_labels(self, meld_label: str) -> str:
        """Converts label from meld to abaw

        Args:
            meld_label (str): Meld label

        Returns:
            str: Abaw label
        """
        meld_labels = {
            "neutral": 0,
            "anger": 1,
            "disgust": 2,
            "fear": 3,
            "joy": 4,
            "sadness": 5,
            "surprise": 6,
        }

        return meld_labels[meld_label]

    def prepare_data(self) -> None:
        """Creates windows with `shift`, `max_w_len`, `min_w_len` in the following steps:
        - Read VAD information
        - Converts labels to ABAW label space
        - Splits obtained VAD sequences with `shift`, `max_w_len`, `min_w_len`:
            skips sequence with length less than `min_w_len`
            or
            splits sequence on windows:
                `seg` - pointer of label index where window is started,
                        iterates from 0 to len of sequence with step of `shift`
                `start` - pointer for first frame number of window (vad_start + `seg`)
                `end` - pointer for last frame number of window without last element
                        vad_end if pointer more then length of vad sequence
                        else (vad_start + `seg` + `max_w_len`)

                if length of window is less than `min_w_len`
                forms window from the end of sequence:
                    `start` - vad_start if `max_w_len` greater than length of window
                              (it means that in sequence there is only one segment with length less than `max_w_len`)
                              else vad_end - `max_w_len`
                    `end` - last frame number (vad_end)
        """
        shift = round(self.shift * self.sr)
        max_w_len = round(self.max_w_len * self.sr)
        min_w_len = round(self.min_w_len * self.sr)

        labs = pd.read_csv(self.labels_file_path, sep=",").to_dict("records")

        vad_info = {}
        with open(self.vad_file_path, "rb") as handle:
            vad_info = pickle.load(handle)

        for lab in labs:
            timings = []

            fn = "dia{0}_utt{1}.wav".format(lab["Dialogue_ID"], lab["Utterance_ID"])
            if "dia125_utt3" in fn:  # filters broken file
                continue

            if fn not in vad_info:  # filters not existing files
                continue

            speech_segments = vad_info[fn]
            for s in speech_segments:
                s_len = s["end"] - s["start"]
                if s_len < min_w_len:  # less than min_w_len
                    continue

                for seg in range(0, s_len, shift):
                    start = s["start"] + seg
                    end = min(
                        s["end"], s["start"] + seg + max_w_len
                    )  # no more than s[end]

                    if (
                        end - start < min_w_len
                    ):  # if less than min_w_len: get last -max_w_len elements
                        start = max(
                            s["start"], s["end"] - max_w_len
                        )  # 0 or frame[-max_w_len] - # no less than s[start]
                        end = s["end"]

                    exprl = self.meld_to_abaw_labels(lab["Emotion"])
                    if exprl > self.num_classes - 1:
                        continue

                    timings.append(
                        {
                            "wav_filename": fn,
                            "start_t": start / self.sr,
                            "end_t": end / self.sr,
                            "start_f": start,
                            "end_f": end,
                            "expr": exprl,
                        }
                    )

            timings = [dict(t) for t in {tuple(d.items()) for d in timings}]
            expr_processed_labs = [t["expr"] for t in timings]

            self.meta.extend(timings)
            self.expr_labels.extend(expr_processed_labs)

        self.expr_labels_counts = np.unique(
            np.asarray(self.expr_labels), return_counts=True
        )[1]
        if self.num_classes > 7:
            self.expr_labels_counts = np.append(self.expr_labels_counts, 0)

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
            "filename": data["wav_filename"],
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
    for ds in ["train", "dev", "test"]:
        md = MeldDataset(
            audio_root=os.path.join(c_config["MELD_WAV_ROOT"], ds),
            labels_file_path=os.path.join(
                c_config["MELD_LABELS_ROOT"], "{0}_sent_emo.csv".format(ds)
            ),
            vad_file_path=os.path.join(
                c_config["MELD_VAD_ROOT"], "vad_{0}_vocals_16000.pickle".format(ds)
            ),
            shift=2,
            min_w_len=2,
            max_w_len=4,
        )

        dl = torch.utils.data.DataLoader(md, batch_size=8, shuffle=False, num_workers=8)

        for d in dl:
            pass

        print("OK")
