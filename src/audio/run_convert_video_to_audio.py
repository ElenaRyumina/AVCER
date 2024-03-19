"""
This is the script for extracting audio from video with/without filtering speech.
"""

import os
import wave
import shutil
import subprocess

import sox
from tqdm import tqdm


def convert_without_filtering(
    inp_path: str, out_path: str, checking: bool = True
) -> None:
    """Convert video to audio using ffmpeg

    Args:
        inp_path (str): Input file path
        out_path (str): Output file path
        checking (bool, optional): Used for checking paths of the ffmpeg command. Defaults to True.
    """
    out_dirname = os.path.dirname(out_path)
    os.makedirs(out_dirname, exist_ok=True)

    # sample rate 16000
    command = f"ffmpeg -y -i {inp_path} -async 1 -vn -acodec pcm_s16le -ar 16000 -ac 1 {out_path}"

    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)


def convert_with_filtering(inp_path: str, out_path: str, checking: bool = True) -> None:
    """Extract speech from the video file using Spleeter and ffmpeg

    Args:
        inp_path (str): Input file path
        out_path (str): Output file path
        checking (bool, optional): Used for checking paths of the spleeter/ffmpeg commands. Defaults to True.
    """
    out_dirname = os.path.dirname(out_path)
    os.makedirs(out_dirname, exist_ok=True)

    # 44100 for spleeter
    command = (
        f"ffmpeg -y -i {inp_path} -async 1 -vn -acodec pcm_s16le -ar 44100 {out_path}"
    )

    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

    inp_duration = sox.file_info.duration(out_path)

    # extract speech using spleeter
    command = f"spleeter separate -o {out_dirname} {out_path} -d 1620"  # maximum length in seconds
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, env=os.environ.copy()
        )

    spleeter_duration = sox.file_info.duration(out_path)

    # convert 44100 to 16000
    command = "ffmpeg -y -i {0} -async 1 -ar 16000 -ac 1 {1}".format(
        os.path.join(
            out_dirname, os.path.basename(out_path).split(".")[0], "vocals.wav"
        ),
        out_path,
    )

    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

        shutil.rmtree(
            os.path.join(out_dirname, os.path.basename(out_path).split(".")[0])
        )

    # check results for errors
    final_duration = sox.file_info.duration(out_path)
    if (abs(inp_duration - spleeter_duration) < 1e-4) and (
        abs(inp_duration - final_duration) < 1e-4
    ):
        pass
    else:
        print(f"Error {inp_path}")
        print(inp_duration, spleeter_duration, final_duration)


def convert_video_to_audio(
    files_root: str,
    db: str,
    wavs_root: str = "wavs",
    vocals_root: str = "vocals",
    filtering: bool = False,
    checking: bool = True,
) -> None:
    """Loops through the directory, and extract speech from each video file using Spleeter and ffmpeg.

    Args:
        files_root (str): Input directory
        db (str): Database: can be 'meld' or 'abaw'
        wavs_root (str, optional): Wavs output path. Defaults to 'wavs'.
        vocals_root (str, optional): Vocals output path. Defaults to 'vocals'.
        filtering (bool, optional): Apply spleeter or not. Defaults to False.
        checking (bool, optional): Used for checking paths of the spleeter/ffmpeg commands. Defaults to True.
    """
    # run on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ds_name = os.path.basename(files_root) if db.lower() == "meld" else ""
    files_root = os.path.dirname(files_root) if db.lower() == "meld" else files_root

    out_wavs_root = os.path.join(os.path.dirname(files_root), wavs_root, ds_name)
    out_vocals_root = os.path.join(os.path.dirname(files_root), vocals_root, ds_name)

    for fn in tqdm(os.listdir(os.path.join(files_root, ds_name))):
        if fn.startswith(".") or "dia125_utt3.mp4" in fn:
            continue

        convert_without_filtering(
            inp_path=os.path.join(files_root, ds_name, fn),
            out_path=os.path.join(
                out_wavs_root, fn.replace("mp4", "wav").replace("avi", "wav")
            ),
            checking=checking,
        )

        if filtering:
            convert_with_filtering(
                inp_path=os.path.join(files_root, ds_name, fn),
                out_path=os.path.join(
                    out_vocals_root, fn.replace("mp4", "wav").replace("avi", "wav")
                ),
                checking=checking,
            )


if __name__ == "__main__":
    files_root = "/"  # TODO
    convert_video_to_audio(
        files_root=files_root, db="abaw", filtering=True, checking=False
    )

    files_root = "/"  # TODO
    convert_video_to_audio(
        files_root=files_root, db="abaw", filtering=True, checking=False
    )

    files_root = "/"  # TODO
    for d in ["train", "dev", "test"]:
        convert_video_to_audio(
            files_root=os.path.join(files_root, d),
            db="meld",
            filtering=True,
            checking=False,
        )
