"""
Model training details are available at scr/audio.

"""

import torch
import os
import numpy as np
import pandas as pd
from data.utils import convert_mp4_to_mp3, pad_wav, pad_wav_zeros
from architectures.audio_8_cl import ExprModelV3
from transformers import (
    AutoFeatureExtractor,
)
import glob
from tqdm import tqdm

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


class EmotionRecognition:
    def __init__(
        self,
        step=2,
        window=4,
        sr=16000,
        device="cuda:0",
        model_params={},
        save_path="",
        padding="",
        flag_save_prob=True,
    ):
        self.model_params = model_params
        self.save_path = save_path
        self.step = step
        self.window = window
        self.sr = sr
        self.device = device
        self.padding = padding
        self.flag_save_prob = flag_save_prob
        self.load_models()

    def predict_emotion(self, path, fps):
        prob = self.load_audio_features(path, fps)
        return prob

    def load_models(self):
        self.load_audio_model()

    def load_audio_model(self):
        path_audio_model = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.processor = AutoFeatureExtractor.from_pretrained(path_audio_model)
        self.audio_model = self.model_params["model_cls"].from_pretrained(
            path_audio_model
        )
        self.audio_model.load_state_dict(
            torch.load(
                os.path.join(
                    self.model_params["root_path"],
                    "epoch_{}.pth".format(self.model_params["epoch"]),
                )
            )["model_state_dict"]
        )
        self.audio_model.to(self.device).eval()

    def load_audio_features(self, path, fps):

        window_a = self.window * self.sr
        step_a = int(self.step * self.sr)

        wav = convert_mp4_to_mp3(path, self.sr)

        probs = []
        framess = []

        for start_a in range(0, len(wav) + 1, step_a):
            end_a = min(start_a + window_a, len(wav))
            # if end_a - start_a < step_a and start_a != 0:
            #     break
            a_fss_chunk = wav[start_a:end_a]
            if self.padding == "mean" or self.padding == "constant":
                a_fss = pad_wav_zeros(a_fss_chunk, window_a, mode=self.padding)
            elif self.padding == "repeat":
                a_fss = pad_wav(a_fss_chunk, window_a)
            a_fss = torch.unsqueeze(a_fss, 0)
            a_fss = self.processor(a_fss, sampling_rate=self.sr)
            a_fss = a_fss["input_values"][0]
            a_fss = torch.from_numpy(a_fss)
            with torch.no_grad():
                prob = self.audio_model(a_fss.to(self.device))
            prob = prob.cpu().numpy()
            frames = [
                str(i).zfill(6) + ".jpg"
                for i in range(
                    round(start_a / self.sr * fps), round(end_a / self.sr * fps + 1)
                )
            ]
            probs.extend([prob] * len(frames))
            framess.extend(frames)

        if len(probs[0]) == 7:
            emo_ABAW = [
                "Neutral",
                "Anger",
                "Disgust",
                "Fear",
                "Happiness",
                "Sadness",
                "Surprise",
            ]
        else:
            emo_ABAW = [
                "Neutral",
                "Anger",
                "Disgust",
                "Fear",
                "Happiness",
                "Sadness",
                "Surprise",
                "Other",
            ]

        df = pd.DataFrame(np.array(probs), columns=emo_ABAW)
        df["frames"] = framess

        if self.flag_save_prob:
            save_path = os.path.join(self.save_path, self.model_params["model_name"])
            os.makedirs(save_path, exist_ok=True)
            name_video = os.path.basename(path[:-4])
            if name_video == "135-24-1920x1080":
                name_video = "135-24-1920x1080_left"
            elif name_video == "6-30-1920x1080":
                name_video = "6-30-1920x1080_right"
            df.to_csv(os.path.join(save_path, "{}.csv".format(name_video)), index=False)

        return df


def preprocess_audio_and_predict(
    path_video="",
    path_weights="",
    save_path="src/pred_results/C-EXPR-DB",
    fps=25,
    step=0.5,
    padding="mean",
    flag_save_prob=False,
    window=4,
    sr=16000,
    device="cuda:0",
):

    model_params = {
        "model_name": "FLW-ExprModelV3-2024.03.02-11.42.11",
        "model_cls": ExprModelV3,
        "epoch": 63,
    }
    model_params["root_path"] = os.path.join(path_weights, model_params["model_name"])
    audio_ER = EmotionRecognition(
        step=step,
        window=window,
        sr=sr,
        device=device,
        model_params=model_params,
        save_path=save_path,
        padding=padding,
        flag_save_prob=flag_save_prob,
    )
    df_pred = audio_ER.predict_emotion(path_video, fps)

    return df_pred


# Example usage:
if __name__ == "__main__":

    flag_save_prob = True

    for padding in ["constant", "mean", "repeat"]:

        for step in [0.5]:

            folder_models = "C:/Work/ABAW_audio_models/"

            parameters = [
                {
                    "model_name": "CELSWa-ExprModelV3-2024.02.27-20.52.14",
                    "model_cls": ExprModelV3,
                    "epoch": 93,
                },
                {
                    "model_name": "CELSW-ExprModelV3-2024.02.28-10.33.12",
                    "model_cls": ExprModelV3,
                    "epoch": 85,
                },
                {
                    "model_name": "FLW-ExprModelV3-2024.03.01-20.26.51",
                    "model_cls": ExprModelV3,
                    "epoch": 26,
                },
                {
                    "model_name": "FLW-ExprModelV3-2024.03.02-11.42.11",
                    "model_cls": ExprModelV3,
                    "epoch": 63,
                },
            ]

            print("C-EXPR-DB", step)

            save_path = "src/pred_results/C-EXPR-DB/audio_{}_{}/".format(padding, step)

            path_videos = glob.glob("C:/Work/Datasets/C-EXPR-DB/*")
            path_videos = [i for i in path_videos if i.endswith(".mp4")]

            df_fps = pd.read_csv("D:/work/ABAW2024/C-EXPR-DB_data.csv")

            filenames_frames = df_fps.name_video.tolist()
            fpss = df_fps.frame_rate.values.tolist()

            for model_params in parameters:
                print(model_params["model_name"])
                model_params["root_path"] = os.path.join(
                    folder_models, model_params["model_name"]
                )
                audio_ER = EmotionRecognition(
                    step=step,
                    window=4,
                    sr=16000,
                    device="cuda:0",
                    model_params=model_params,
                    save_path=save_path,
                    padding=padding,
                    flag_save_prob=flag_save_prob,
                )
                for name_video in tqdm(path_videos):
                    fps = fpss[filenames_frames.index(name_video.split("\\")[-1])]
                    _ = audio_ER.predict_emotion(name_video, fps)

            print("AFEW", step)

            save_path = "src/pred_results/AFEW/audio_{}_{}/".format(padding, step)

            path_videos = glob.glob("D:/Databases/AFEW/Val_AFEW/*/*")
            path_videos = [i for i in path_videos if i.endswith(".avi")]

            df_fps = pd.read_csv("D:/work/ABAW2024/AFEW_data.csv")

            filenames_frames = df_fps.name_video.tolist()
            fpss = df_fps.frame_rate.values.tolist()

            for model_params in parameters:
                print(model_params["model_name"])
                model_params["root_path"] = os.path.join(
                    folder_models, model_params["model_name"]
                )
                audio_ER = EmotionRecognition(
                    step=step,
                    window=4,
                    sr=16000,
                    device="cuda:0",
                    model_params=model_params,
                    save_path=save_path,
                    padding=padding,
                    flag_save_prob=flag_save_prob,
                )
                for name_video in tqdm(path_videos):
                    # print(name_video)
                    fps = fpss[filenames_frames.index(name_video.split("\\")[-1])]
                    _ = audio_ER.predict_emotion(name_video, fps)

            print("ABAW", step)

            name_videos = [
                "117",
                "118-30-640x480",
                "118",
                "119",
                "120-30-1280x720",
                "120",
                "121-24-1920x1080",
                "121",
                "122-60-1920x1080-1",
                "122-60-1920x1080-2",
                "122-60-1920x1080-3",
                "122-60-1920x1080-4",
                "122",
                "123-25-1920x1080",
                "123",
                "124-30-720x1280",
                "125-25-1280x720",
                "125",
                "126",
                "127-30-1280x720",
                "127",
                "128-24-1920x1080",
                "128",
                "129-24-1280x720",
                "130",
                "131-30-1920x1080",
                "131",
                "132",
                "133",
                "134",
                "135",
                "136",
                "137-30-1920x1080",
                "137",
                "138",
                "139",
                "140-30-632x360",
                "140",
                "141",
                "143",
                "144",
                "146",
                "147",
                "148",
                "149",
                "150",
                "151",
                "153",
                "154",
                "155",
                "156",
                "157",
                "158",
                "159",
                "160",
                "161",
                "162",
                "163",
                "164",
                "165",
                "24-30-1920x1080-1",
                "24-30-1920x1080-2",
                "28-30-1280x720-2",
                "282",
                "45-24-1280x720",
                "82-25-854x480",
                "video34",
                "video73",
                "135-24-1920x1080",
                "6-30-1920x1080",
            ]

            save_path = "src/pred_results/ABAW/audio_{}_{}/".format(padding, step)
            path_video = "C:/Work/Datasets/ABAW/Video/"

            df_fps = pd.read_csv("D:/work/ABAW2024/videos_frame_rate.txt")
            name_videos_true = [
                i for i in df_fps.filename.tolist() if i[:-4] in name_videos
            ]
            filenames_fps = df_fps.filename.tolist()
            fpss = df_fps.frame_rate.values.tolist()

            for model_params in parameters:
                print(model_params["model_name"])
                model_params["root_path"] = os.path.join(
                    folder_models, model_params["model_name"]
                )
                audio_ER = EmotionRecognition(
                    step=step,
                    window=4,
                    sr=16000,
                    device="cuda:0",
                    model_params=model_params,
                    save_path=save_path,
                    padding=padding,
                    flag_save_prob=flag_save_prob,
                )
                for name_video in tqdm(name_videos_true):
                    fps = fpss[filenames_fps.index(name_video.split("\\")[-1])]
                    _ = audio_ER.predict_emotion(
                        os.path.join(path_video, name_video), fps
                    )
