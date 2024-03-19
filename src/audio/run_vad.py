import os
import pickle

from tqdm import tqdm
import torch


def main(path_to_audio: str, output_path: str, sr: int = 16000) -> None:
    """Extracts timings of speech in audio into pickle

    Args:
        path_to_audio (str): Path to directory with audio
        output_path (str): Path to resulted pickle file
        sr (int): Sample rate of audio. Defaults to 16000.
    """
    use_onnx = False
    output_path, _ = output_path.split(".pickle")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=True,
        onnx=use_onnx,
    )

    (get_speech_timestamps, _, read_audio, _, _) = utils

    res = {}
    for f in tqdm(os.listdir(path_to_audio)):
        wav = read_audio(os.path.join(path_to_audio, f), sampling_rate=sr)
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)
        res[f] = speech_timestamps

    with open("{0}_{1}.pickle".format(output_path, sr), "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    for fold in ["wavs", "vocals"]:
        path_to_images = "/{0}/".format(fold)  # TODO
        path_to_pickle = "/vad_{0}.pickle".format(fold)  # TODO
        main(path_to_images, path_to_pickle)

        path_to_images = "/{0}/".format(fold)  # TODO
        path_to_pickle = "/"  # TODO
        for ds in ["train", "dev", "test"]:
            main(
                os.path.join(path_to_images, ds),
                os.path.join(path_to_pickle, "vad_{0}_{1}.pickle".format(ds, fold)),
            )
