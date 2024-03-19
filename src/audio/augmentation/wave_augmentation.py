import numpy as np
import random

import torch
import torchaudio


class PolarityInversion(torch.nn.Module):
    """Inverses all values of wave"""

    def __init__(self) -> None:
        super(PolarityInversion, self).__init__()

    def forward(self, wave: torch.Tensor) -> torch.tensor:
        """Inverses all values of wave

        Args:
            wave (torch.Tensor): Input audio tensor

        Returns:
            torch.tensor: Inversed audio tensor
        """
        wave = torch.neg(wave)
        return wave


class WhiteNoise(torch.nn.Module):
    """Adds white noise to audio tensor

    Args:
        min_snr (float, optional): Minimum signal to noise ration value. Defaults to 0.0001.
        max_snr (float, optional): Maximum signal to noise ration value. Defaults to 0.005.
    """

    def __init__(self, min_snr: float = 0.0001, max_snr: float = 0.005) -> None:
        super(WhiteNoise, self).__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Adds white noise to audio tensor

        Args:
            audio (torch.Tensor): Input audio tensor

        Returns:
            torch.Tensor: Noised audio tensor
        """
        std = torch.std(audio).numpy()
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)
        noise = np.random.normal(0.0, noise_std, size=audio.shape).astype(np.float32)

        return audio + torch.Tensor(noise)


class SoxEffect(torch.nn.Module):
    """Applies sox effects to given audio tensor

    Args:
        effects (list[list[str]]): List of sox effects
        sr (int, optional): Sample rate of audio. Defaults to 16000.
    """

    def __init__(self, effects: list[list[str]], sr: int = 16000) -> None:
        super(SoxEffect, self).__init__()
        self.effects = effects
        self.sr = sr

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Applies sox effects to given audio tensor

        Args:
            wave (torch.Tensor): Input audio tensor

        Returns:
            torch.Tensor: Audio tensor with sox effects
        """
        wave, sr = torchaudio.sox_effects.apply_effects_tensor(
            wave, self.sr, self.effects
        )
        return wave


class Gain(torch.nn.Module):
    """Changes volume of audio on random value in specified range.

    Args:
        min_gain (float, optional): Minimum gain value. Defaults to -20.0.
        max_gain (float, optional): Maximum gain value. Defaults to -1.
    """

    def __init__(self, min_gain: float = -20.0, max_gain: float = -1) -> None:
        super(Gain, self).__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Changes volume of audio on random value in specified range.

        Args:
            audio (torch.Tensor): Input audio tensor

        Returns:
            torch.Tensor: Audio tensor with increased volume
        """
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = torchaudio.transforms.Vol(gain, gain_type="db")(wave)
        return audio


class RandomChoice(torch.nn.Module):
    """Chooses randomly one transform from list of transforms, and applies to tensor

    Args:
        transforms (list of ``Transform`` objects): list of transform objects
    """

    def __init__(self, transforms) -> None:
        super(RandomChoice, self).__init__()
        self.transforms = transforms

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Picks and applies random transformation on audio tensor

        Args:
            wave (torch.Tensor): Input audio tensor

        Returns:
            torch.Tensor: Transformed audio tensor
        """
        t = random.choice(self.transforms)
        return t(wave)


class ResampleAudio(torch.nn.Module):
    """Converts sample rate of audio tensor using torchaudio.transforms.Resample if sample rates are different

    Args:
        orig_sr (int, optional): Original sample rate. Defaults to 32000.
        new_sr (int, optional): New sample rate. Defaults to 16000.
    """

    def __init__(self, orig_sr: int = 32000, new_sr: int = 16000) -> None:
        super(ResampleAudio, self).__init__()
        if orig_sr != new_sr:
            self.transforms = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=new_sr
            )
        else:
            self.transforms = None

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Converts sample rate of audio tensor using transform

        Args:
            wave (torch.Tensor): Input audio tensor

        Returns:
            torch.Tensor: Converted audio tensor
        """
        return self.transforms(wave) if self.transforms else wave
