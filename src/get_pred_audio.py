import pandas as pd
import os
import numpy as np
import pickle
from tqdm import tqdm
from visualization.visualize import plot_conf_matrix
from sklearn.metrics import confusion_matrix
from data.utils import (
    metrics,
    softmax,
    get_compound_expression,
    save_txt,
    get_image_location,
)


def get_metrics(trues, predictions, corpus, name_model, modality):

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

    a_pred = np.argmax(predictions, axis=-1).astype("int32")
    uar_a, acc_a, f1_a, precision_a, mean_a = metrics(trues, a_pred)
    cm_dyn = confusion_matrix(
        trues, a_pred, labels=[l for l in range(len(name_emo[:-1]))]
    )
    save_path = "src/pred_results/confusion_matrices/"

    os.makedirs(save_path, exist_ok=True)
    plot_conf_matrix(
        cm=cm_dyn,
        title="Audio model (). {0}. UAR = {1:.2f}%".format(corpus, uar_a * 100),
        labels=name_emo[:-1],
        save_path=os.path.join(save_path, f"{corpus}_{modality}_{name_model}.pdf"),
    )

    dict_metrics = {
        "uar_a": uar_a,
        "acc_a": acc_a,
        "f1_a": f1_a,
        "precision_a": precision_a,
        "mean_a": mean_a,
    }

    print(dict_metrics)

    save_path = "src/pred_results/metrics_dicts/"
    os.makedirs(save_path, exist_ok=True)

    filename = os.path.join(
        save_path, f"{corpus}_metrics_dict_{modality}_{name_model}.pickle"
    )
    pickle.dump(dict_metrics, open(filename, "wb"))


def get_abaw_pred(path_ann, path_pred, ann_files, corpus, name_model, modality):
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
    trues = []
    preds_audio = []

    for curr_ann_file in tqdm(ann_files):
        curr_df = pd.read_csv(os.path.join(path_ann, curr_ann_file))
        curr_pred_audio = pd.read_csv(
            os.path.join(path_pred, curr_ann_file[:-4]) + ".csv"
        )
        curr_pred_audio = curr_pred_audio.groupby(["frames"]).mean().reset_index()
        need_index = curr_df.loc[~curr_df["Neutral"].isin([-1, 7])].index
        true = curr_df[curr_df.index.isin(need_index)].Neutral.values.tolist()
        curr_pred_audio = curr_pred_audio[curr_pred_audio.index.isin(need_index)][
            name_emo[:-1]
        ].values
        curr_pred_audio = softmax(curr_pred_audio[:, :7]).tolist()
        if len(curr_pred_audio) < len(true):
            last_pred_audio = curr_pred_audio[-1]
            curr_pred_audio += [last_pred_audio] * (len(true) - len(curr_pred_audio))
        preds_audio.extend(curr_pred_audio)
        trues.extend(true)

    get_metrics(trues, preds_audio, corpus, name_model, modality)


def get_afew_pred(path_pred, corpus, name_model, modality):
    df_emotion = pd.read_csv("D:/work/ABAW2024/AFEW_data.csv")
    name_videos = [i[:-4] for i in df_emotion.name_video]
    emotion = df_emotion.emotion.values

    # audio model prediction order
    dict_emo = {
        "Angry": 1,
        "Disgust": 2,
        "Fear": 3,
        "Happy": 4,
        "Neutral": 0,
        "Sad": 5,
        "Surprise": 6,
    }
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

    trues = []
    preds_audio = []

    for curr_video, curr_emotion in tqdm(zip(name_videos, emotion)):
        curr_pred_audio = pd.read_csv(
            os.path.join(path_pred, curr_video) + ".csv"
        ).dropna()
        curr_pred_audio = curr_pred_audio.groupby(["frames"]).mean().reset_index()
        curr_pred_audio = softmax(curr_pred_audio[name_emo[:-1]].values)

        true = dict_emo[curr_emotion]
        pred_audio = np.mean(curr_pred_audio, axis=0)

        trues.append(true)
        preds_audio.append(pred_audio)

    get_metrics(trues, preds_audio, corpus, name_model, modality)


def get_c_expr_db_pred(
    prediction_file_format,
    path_pred,
    name_videos,
    name_model,
    modality,
    ce_weights_type,
):

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

    df_prediction_file_format = pd.read_csv(prediction_file_format)
    df_prediction_file_format["curr_video"] = [
        i.split("/")[0] for i in df_prediction_file_format.image_location
    ]

    preds_audio = []
    image_locations = []
    for curr_video in tqdm(name_videos):
        curr_pred_audio = pd.read_csv(
            os.path.join(path_pred, curr_video) + ".csv"
        ).dropna()
        curr_pred_audio = curr_pred_audio.groupby(["frames"]).mean().reset_index()
        curr_pred_audio = curr_pred_audio.rename(columns={"frames": "image_location"})
        curr_pred_audio["image_location"] = [
            get_image_location(curr_video, i) for i in curr_pred_audio.image_location
        ]
        image_location = curr_pred_audio.image_location.tolist()
        image_location_true = df_prediction_file_format[
            df_prediction_file_format.curr_video == curr_video
        ].image_location.tolist()
        need_image_location = list(set(image_location) & set(image_location_true))
        need_image_location = sorted(need_image_location)
        curr_pred_audio = curr_pred_audio[
            curr_pred_audio.image_location.isin(need_image_location)
        ][name_emo[:-1]].values
        if len(image_location_true) > len(curr_pred_audio):
            last_pred_audio = curr_pred_audio[-1]
            curr_pred_audio = np.vstack(
                (
                    curr_pred_audio,
                    [last_pred_audio]
                    * (len(image_location_true) - len(curr_pred_audio)),
                )
            )
        curr_pred_audio = softmax(curr_pred_audio)
        preds_audio.extend(curr_pred_audio)
        image_locations.extend(image_location_true)

    com_emo = {
        "Fearfully Surprised": [3, 6],
        "Happily Surprised": [4, 6],
        "Sadly Surprised": [5, 6],
        "Disgustedly Surprised": [2, 6],
        "Angrily Surprised": [1, 6],
        "Sadly Fearful": [3, 5],
        "Sadly Angry": [1, 5],
    }

    dict_weights = {
        1: 5,
        2: 6,
        3: 5,
        4: 6,
        5: 4,
        6: 2,
    }
    audio_prob_raw = np.array(preds_audio)
    audio_prob = get_compound_expression(
        audio_prob_raw, com_emo, dict_weights, ce_weights_type
    )

    save_path = "src/pred_results/DF_C_EXPR_DB/"
    os.makedirs(save_path, exist_ok=True)

    audio_pred = np.argmax(audio_prob[:, :7], axis=1)

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
        image_locations,
        audio_pred,
        os.path.join(
            save_path,
            f"C_EXPR_DB_{modality}_{name_model}_ce_type_{ce_weights_type}.txt",
        ),
    )


if __name__ == "__main__":

    np.random.seed(seed=42)

    path_ann = "C:/Work/Datasets/ABAW/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/"
    path_pred_ABAW = "src/pred_results/ABAW/"
    path_pred_AFEW = "src/pred_results/AFEW/"
    path_pred_C = "src/pred_results/C-EXPR-DB/"

    ce_weights_types = [True, False]

    ann_files = os.listdir(path_ann)

    for name_folder in [
        "audio_constant_0.5",
        "audio_constant_1",
        "audio_constant_2",
        "audio_constant_3",
        "audio_constant_4",
        "audio_mean_0.5",
        "audio_mean_1",
        "audio_mean_2",
        "audio_mean_3",
        "audio_mean_4",
        "audio_repeat_0.5",
        "audio_repeat_1",
        "audio_repeat_2",
        "audio_repeat_3",
        "audio_repeat_4",
    ]:

        name_models = os.listdir(os.path.join(path_pred_ABAW, name_folder))

        for name_model in name_models:
            cur_path_pred = os.path.join(path_pred_ABAW, name_folder, name_model)
            print(name_folder, name_model)

            print("ABAW")
            get_abaw_pred(
                path_ann, cur_path_pred, ann_files, "ABAW", name_model, name_folder
            )

            print("AFEW")
            cur_path_pred = os.path.join(path_pred_AFEW, name_folder, name_model)
            get_afew_pred(cur_path_pred, "AFEW", name_model, name_folder)

            print("C-EXPR-DB")
            prediction_file_format = "C:/Work/Datasets/ABAW/prediction_files_format/CVPR_6th_ABAW_CE_test_set_sample.txt"
            df_name_videos = pd.read_csv("D:/work/ABAW2024/C-EXPR-DB_data.csv")
            name_videos = [i[:-4] for i in df_name_videos.name_video]
            cur_path_pred = os.path.join(path_pred_C, name_folder, name_model)

            for ce_weights_type in ce_weights_types:
                get_c_expr_db_pred(
                    prediction_file_format,
                    cur_path_pred,
                    name_videos,
                    name_model,
                    name_folder,
                    ce_weights_type,
                )
