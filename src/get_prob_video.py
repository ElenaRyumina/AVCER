"""
Model training details are available at scr/video.

"""

import torch
import cv2
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import glob

from data.utils import pth_processing, get_heatmaps
from architectures.video import ResNet50, LSTMPyTorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

path_static = "src/weights/FER_static_ResNet50_AffectNet.pt"
pth_model_static = ResNet50(7, channels=3)
pth_model_static.load_state_dict(torch.load(path_static))
pth_model_static.to(device).eval()

activations = {}


def get_activations(name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


gradients = {}


def get_gradients(name):
    def hook(model, input, output):
        gradients[name] = output

    return hook


pth_model_static.layer4.register_full_backward_hook(get_gradients("layer4"))
pth_model_static.layer4.register_forward_hook(get_activations("layer4"))
pth_model_static.fc1.register_forward_hook(get_activations("features"))

path_dynamic = "src/weights/FER_dinamic_LSTM_Aff-Wild2.pt"
pth_model_dynamic = LSTMPyTorch()
pth_model_dynamic.load_state_dict(torch.load(path_dynamic))
pth_model_dynamic.to(device).eval()

DICT_EMO_VIDEO = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
}


def preprocess_video_and_predict(
    path_images="",
    save_path="",
    fps=30,
    total_frames=[],
    flag_save_prob=False,
    flag_heatmaps=False,
    model_heatmaps=None,
):

    step = round((5 * fps) / 25)

    all_frames = os.listdir(os.path.join(path_images, "00"))

    count_frame = 1
    last_output = None
    cur_face = None
    probs_dynamic = []
    probs_static = []
    frames = []
    lstm_features = []

    zeros = np.zeros((1, 7))

    for curr_idx_frame in range(0, total_frames):

        curr_frame = str(curr_idx_frame).zfill(6) + ".jpg"
        if curr_frame in all_frames:
            frame = cv2.imread(os.path.join(path_images, "00", curr_frame))
            cur_face = frame.copy()
            cur_face = cv2.cvtColor(cur_face, cv2.COLOR_BGR2RGB)

            cur_face_copy = pth_processing(Image.fromarray(cur_face))

            if flag_heatmaps:
                prediction = F.softmax(
                    pth_model_static(cur_face_copy.to(device)), dim=1
                )
            else:
                with torch.no_grad():
                    prediction = F.softmax(
                        pth_model_static(cur_face_copy.to(device)), dim=1
                    )

            output_s = prediction.clone()
            output_s = output_s.cpu().detach().numpy()

            if curr_idx_frame % step == 0:
                features = F.relu(activations["features"]).cpu().detach().numpy()

                if len(lstm_features) == 0:
                    lstm_features = [features] * 10
                else:
                    lstm_features = lstm_features[1:] + [features]

                lstm_f = torch.from_numpy(np.vstack(lstm_features))
                lstm_f = torch.unsqueeze(lstm_f, 0)

                with torch.no_grad():
                    output_d = (
                        pth_model_dynamic(lstm_f.to(device)).cpu().detach().numpy()
                    )
                last_output = output_d

                if flag_heatmaps:
                    if model_heatmaps == "dynamic":
                        max_idx = np.argmax(output_d, axis=1)[0]
                    elif model_heatmaps == "static":
                        max_idx = np.argmax(output_s, axis=1)[0]

                    prediction[:, max_idx].backward()
                    heatmap = get_heatmaps(
                        gradients,
                        activations,
                        "layer4",
                        cur_face,
                        use_rgb=False,
                        image_weight=0.8,
                    )
                    save_path_full = os.path.join(
                        save_path,
                        os.path.basename(path_images),
                        f"heatmaps_{model_heatmaps}",
                    )
                    os.makedirs(save_path_full, exist_ok=True)
                    cv2.imwrite(os.path.join(save_path_full, curr_frame), heatmap)

                gradients.clear()
                activations.clear()

            else:
                if last_output is not None:
                    output_d = last_output

                elif last_output is None:
                    output_d = zeros

            probs_static.append(output_s[0])
            probs_dynamic.append(output_d[0])
            frames.append(count_frame)

        else:
            lstm_features = []
            if last_output is not None:
                probs_static.append(probs_static[-1])
                probs_dynamic.append(probs_dynamic[-1])
                frames.append(count_frame)

            elif last_output is None:
                probs_static.append(zeros[0])
                probs_dynamic.append(zeros[0])
                frames.append(count_frame)

        count_frame += 1

    df_dynamic = pd.DataFrame(
        np.array(probs_dynamic), columns=list(DICT_EMO_VIDEO.values())
    )
    df_static = pd.DataFrame(
        np.array(probs_static), columns=list(DICT_EMO_VIDEO.values())
    )

    if flag_save_prob:
        os.makedirs(save_path, exist_ok=True)
        df_dynamic.to_csv(
            os.path.join(
                save_path, "dynamic__{}.csv".format(os.path.basename(path_images))
            ),
            index=False,
        )
        df_static.to_csv(
            os.path.join(
                save_path, "static__{}.csv".format(os.path.basename(path_images))
            ),
            index=False,
        )

    return df_dynamic, df_static


if __name__ == "__main__":

    flag_save_prob = True

    # AFEW

    path_save = "src/pred_results/AFEW/video_2/"
    path_videos = glob.glob("C:/Work/Faces/AFEW_faces/Val_AFEW/VIDEO/*/*")

    df_fps = pd.read_csv("D:/work/ABAW2024/AFEW_data.csv")

    filenames_frames = [i[:-4] for i in df_fps.name_video]
    framess = df_fps.total_frame.values.tolist()
    fpss = df_fps.frame_rate.values.tolist()

    for name_video in tqdm(path_videos):
        fps = fpss[filenames_frames.index(name_video.split("\\")[-1])]
        frames = framess[filenames_frames.index(name_video.split("\\")[-1])]
        _, _ = preprocess_video_and_predict(
            path_video=os.path.join(name_video),
            save_path=path_save,
            fps=fps,
            total_frames=frames,
            flag_save_prob=flag_save_prob,
            flag_heatmaps=False,
            model_heatmaps="static",
        )

    # ABAW

    path_videos = "C:/Work/Faces/ABAW_faces/"
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
        "135-24-1920x1080_left",
        "6-30-1920x1080_right",
    ]
    df_fps = pd.read_csv("D:/work/ABAW2024/videos_frame_rate.txt")
    df_frames = pd.read_csv("D:/work/ABAW2024/counter_frame.csv")
    filenames_fps = [i[:-4] for i in df_fps.filename]
    filenames_frames = [i[:-4] for i in df_frames.name_video]
    framess = df_frames.total_frame.values.tolist()
    fpss = df_fps.frame_rate.values.tolist()

    path_save = "src/pred_results/ABAW/video_2/"

    for name_video in tqdm(name_videos):
        if name_video in ["135-24-1920x1080_left", "6-30-1920x1080_right"]:
            fps = fpss[filenames_fps.index(name_video.split("_")[0].split("\\")[-1])]
            frames = framess[
                filenames_frames.index(name_video.split("_")[0].split("\\")[-1])
            ]
        else:
            fps = fpss[filenames_fps.index(name_video.split("\\")[-1])]
            frames = framess[filenames_frames.index(name_video.split("\\")[-1])]
        _, _ = preprocess_video_and_predict(
            path_video=os.path.join(path_videos, name_video),
            save_path=path_save,
            fps=fps,
            total_frames=frames,
            flag_save_prob=flag_save_prob,
            flag_heatmaps=False,
            model_heatmaps="static",
        )

    # C-EXPR-DB

    path_videos = glob.glob("C:/Work/Faces/C-EXPR-DB_faces/*")

    df_fps = pd.read_csv("D:/work/ABAW2024/C-EXPR-DB_data.csv")

    filenames_frames = [i[:-4] for i in df_fps.name_video]
    framess = df_fps.total_frame.values.tolist()
    fpss = df_fps.frame_rate.values.tolist()

    path_save = "src/pred_results/C-EXPR-DB/video_2/"

    for name_video in tqdm(path_videos):
        fps = fpss[filenames_frames.index(name_video.split("\\")[-1])]
        frames = framess[filenames_frames.index(name_video.split("\\")[-1])]
        _, _ = preprocess_video_and_predict(
            path_video=os.path.join(name_video),
            save_path=path_save,
            fps=fps,
            total_frames=frames,
            flag_save_prob=flag_save_prob,
            flag_heatmaps=False,
            model_heatmaps="static",
        )
