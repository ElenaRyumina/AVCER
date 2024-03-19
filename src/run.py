import os
import time
import numpy as np
import pandas as pd
from typing import Optional
from data.get_face_images import VideoPredictor
from get_prob_video import preprocess_video_and_predict
from get_prob_audio_8_cl import preprocess_audio_and_predict
from visualization.visualize import plot_compound_expression_prediction
from data.utils import softmax, get_compound_expression, save_txt, get_image_location
import argparse

parser = argparse.ArgumentParser(description="run")

parser.add_argument(
    "--path_video", type=str, default="video/", help="Path to a video file"
)
parser.add_argument(
    "--path_save", type=str, default="report/", help="Path to save the results"
)

args = parser.parse_args()


def get_c_expr_db_pred(
    stat_df: pd.DataFrame,
    dyn_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    name_video: str,
    weights_1: list[float],
    weights_2: list[float],
    ce_weights_type: bool,
    ce_mask: bool,
    flag_save_prob: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Predict compound expressions using audio-visual emotional probabilities, optimized weights, and rules.

    Args:
        stat_df (pd.DataFrame): DataFrame containing static visual probabilities.
        dyn_df (pd.DataFrame): DataFrame containing dynamic visual probabilities.
        audio_df (pd.DataFrame): DataFrame containing audio probabilities.
        name_video (str): Name of the video.
        weights_1 (List[float]): List of weights for the Dirichlet-based fusion.
        weights_2 (List[float]): List of weights for the hierarchical-based fusion.
        ce_weights_type (bool): Type of weights for compound expression by Rule 2.
        ce_mask (bool): Type of weights for compound expression by Rule 1.
        flag_save_prob (bool): Flag whether to save predictions of compound expressions to a txt file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]: Predictions for compound expressions,
            and list of image locations.
    """

    # audio model prediction order
    name_emo = [
        "Neutral",
        "Anger",
        "Disgust",
        "Fear",
        "Happiness",
        "Sadness",
        "Surprise",
        "Other",
    ]
    com_emo = {
        "Fearfully Surprised": [3, 6],
        "Happily Surprised": [4, 6],
        "Sadly Surprised": [5, 6],
        "Disgustedly Surprised": [2, 6],
        "Angrily Surprised": [1, 6],
        "Sadly Fearful": [3, 5],
        "Sadly Angry": [1, 5],
    }

    stat_df["image_location"] = [
        f"{name_video}/{str(f+1).zfill(5)}.jpg" for f in stat_df.index
    ]
    dyn_df["image_location"] = [
        f"{name_video}/{str(f+1).zfill(5)}.jpg" for f in dyn_df.index
    ]

    image_location = dyn_df.image_location.tolist()

    stat_df = stat_df[stat_df.image_location.isin(image_location)][name_emo[:-1]].values
    dyn_df = softmax(
        dyn_df[dyn_df.image_location.isin(image_location)][name_emo[:-1]].values
    )

    audio_df = audio_df.groupby(["frames"]).mean().reset_index()
    audio_df = audio_df.rename(columns={"frames": "image_location"})
    audio_df["image_location"] = [
        get_image_location(name_video, i) for i in audio_df.image_location
    ]
    audio_df = softmax(
        audio_df[audio_df.image_location.isin(image_location)][name_emo[:-1]].values
    )

    if len(image_location) > len(audio_df):
        last_pred_audio = audio_df[-1]
        audio_df = np.vstack(
            (audio_df, [last_pred_audio] * (len(image_location) - len(audio_df)))
        )

    predictions = [stat_df, dyn_df, audio_df]
    num_predictions = len(predictions)

    if weights_1:
        final_predictions = predictions[0] * weights_1[0] * weights_2[0]
        for i in range(1, num_predictions):
            final_predictions += predictions[i] * weights_1[i] * weights_2[i]

    else:
        final_predictions = np.sum(predictions, axis=0) / num_predictions

    dict_weights = {
        1: 5,
        2: 6,
        3: 5,
        4: 6,
        5: 4,
        6: 2,
    }

    av_prob = get_compound_expression(
        final_predictions, com_emo, dict_weights, ce_weights_type, ce_mask
    )

    if weights_1:
        vs_prob = get_compound_expression(
            predictions[0] * weights_1[0] * weights_2[0],
            com_emo,
            dict_weights,
            ce_weights_type,
            ce_mask,
        )
        vd_prob = get_compound_expression(
            predictions[1] * weights_1[1] * weights_2[1],
            com_emo,
            dict_weights,
            ce_weights_type,
            ce_mask,
        )
        a_prob = get_compound_expression(
            predictions[2] * weights_1[2] * weights_2[2],
            com_emo,
            dict_weights,
            ce_weights_type,
            ce_mask,
        )
    else:
        vs_prob = get_compound_expression(
            predictions[0], com_emo, dict_weights, ce_weights_type, ce_mask
        )
        vd_prob = get_compound_expression(
            predictions[1], com_emo, dict_weights, ce_weights_type, ce_mask
        )
        a_prob = get_compound_expression(
            predictions[2], com_emo, dict_weights, ce_weights_type, ce_mask
        )

    av_ce = np.argmax(av_prob[:, :7], axis=1)
    vs_ce = np.argmax(vs_prob[:, :7], axis=1)
    vd_ce = np.argmax(vd_prob[:, :7], axis=1)
    a_ce = np.argmax(a_prob[:, :7], axis=1)

    if flag_save_prob:
        save_path = "src/pred_results/DF_C_EXPR_DB/"
        os.makedirs(save_path, exist_ok=True)
        column_names = [
            "image_location",
            "Fearfully_Surprised",
            "Happily_Surprised",
            "Sadly_Surprised",
            "Disgustedly_Surprised",
            "Angrily_Surprised",
            "Sadly_Fearful",
            "Sadly_Angry",
        ]
        save_txt(
            column_names,
            image_location,
            av_ce,
            os.path.join(
                save_path, f"C_EXPR_DB_av_{ce_weights_type}_{ce_mask}_{name_video}.txt"
            ),
        )

    return av_ce, vs_ce, vd_ce, a_ce, image_location


def run_inference(
    path_video: str = "",
    path_save_results: str = "",
    flag_save_prob: bool = False,
    weights_prob_model: Optional[list[float]] = None,
    weights_model: Optional[list[float]] = [1, 1, 1],
    flag_heatmaps: bool = False,
    model_heatmaps: str = "static",
    ce_weights_type: bool = True,
    ce_mask: bool = False,
    flag_save_plot_pred: bool = True,
) -> None:
    """
    Perform inference on a video, including face region detection, emotion prediction using visual and audio models,
    and compound expression prediction.

    Args:
        path_video (str): Path to the input video.
        path_save_results (str): Path to save the results (images, predictions).
        flag_save_prob (bool): Flag indicating whether to save probabilities.
        weights_prob_model (Optional[List[float]]): List of weights for the Dirichlet-based fusion.
        weights_model (Optional[List[float]]): List of weights for the hierarchical-based fusion.
        flag_heatmaps (bool): Flag indicating whether to generate heatmaps.
        model_heatmaps (str): Model for generating heatmaps.
        ce_weights_type (bool): Type of weights for compound expression by Rule 2.
        ce_mask (bool): Type of weights for compound expression by Rule 1.
        flag_save_plot_pred (bool): Flag indicating whether to save the plot predictions.
    """

    start_time = time.time()

    print(f"Face images detection in video: {os.path.basename(path_video)}")
    detect = VideoPredictor()
    detect.process(path_video, path_save_results)

    fps, total_frames = detect.fps, detect.total_frames

    print(f"Emotion prediction using visual models")

    df_probs_dynamic, df_probs_static = preprocess_video_and_predict(
        path_images=os.path.join(path_save_results, os.path.basename(path_video)[:-4]),
        save_path=path_save_results,
        fps=fps,
        total_frames=total_frames,
        flag_save_prob=flag_save_prob,
        flag_heatmaps=flag_heatmaps,
        model_heatmaps=model_heatmaps,
    )

    print(f"Emotion prediction using audio model")

    df_probs_audio = preprocess_audio_and_predict(
        path_video=path_video,
        path_weights="src\weights",
        fps=fps,
        step=0.5,
        padding="mean",
        save_path=path_save_results,
        flag_save_prob=flag_save_prob,
        window=4,
        sr=16000,
        device="cuda:0",
    )

    print(f"Compound expression prediction")

    av_pred_1, vs_pred_1, vd_pred_1, a_pred_1, _ = get_c_expr_db_pred(
        stat_df=df_probs_static,
        dyn_df=df_probs_dynamic,
        audio_df=df_probs_audio,
        name_video=os.path.basename(path_video)[:-4],
        weights_1=weights_prob_model,
        weights_2=weights_model,
        ce_weights_type=ce_weights_type,
        ce_mask=ce_mask,
        flag_save_prob=flag_save_prob,
    )

    end_time = time.time()

    if flag_save_plot_pred:

        preds = {
            "VS": vs_pred_1,
            "VD": vd_pred_1,
            "A": a_pred_1,
            "AV": av_pred_1,
        }

        if ce_mask:
            rule = "Rule 1"
        if ce_weights_type:
            rule = "Rule 2"

        save_path_rule = os.path.join(path_save_results, f"pedicted_CEs_{rule}.jpg")

        plot_compound_expression_prediction(
            preds,
            save_path=save_path_rule,
            title=f"Сompound expressions predicted by models",
        )

        print(
            f"Predictions of compound expression successfully obtained and save as {save_path_rule}"
        )

    print(
        f"Face images are saved in {os.path.join(path_save_results,os.path.basename(path_video)[:-4])}"
    )
    if flag_heatmaps:
        folder_name = f"heatmaps_{model_heatmaps}"
        print(
            f"Heatmap images are saved in {os.path.join(path_save_results, os.path.basename(path_video)[:-4], folder_name)}"
        )
    print(
        f"Real-time factor for compound expression prediction: {((end_time-start_time)/(total_frames/fps)):.2f}"
    )


if __name__ == "__main__":

    path_video = args.path_video
    path_save_results = args.path_save

    weights_av_1 = [
        [
            0.89900098,
            0.10362151,
            0.08577635,
            0.04428126,
            0.89679865,
            0.02656456,
            0.63040305,
        ],
        [
            0.01223291,
            0.21364307,
            0.66688002,
            0.93791526,
            0.0398964,
            0.48670648,
            0.22089692,
        ],
        [
            0.08876611,
            0.68273542,
            0.24734363,
            0.01780348,
            0.06330495,
            0.48672896,
            0.14870002,
        ],
    ]

    run_inference(
        path_video=path_video,
        path_save_results=path_save_results,
        weights_prob_model=weights_av_1,
        flag_save_prob=True,
        flag_heatmaps=False,
        model_heatmaps=None,
        ce_weights_type=False,
        ce_mask=True,
    )
