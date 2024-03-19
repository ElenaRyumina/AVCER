import os
import logging
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils.accuracy_utils import conf_matrix
from visualization.visualize import plot_conf_matrix
from utils.common_utils import create_logger


class ProblemType(Enum):
    """Problem type Enum
    Used in NetTrainer
    """

    CLASSIFICATION: int = 1
    REGRESSION: int = 2


class NetTrainer:
    """Trains the model
    - Performs logging:
         Logs general information (epoch number, phase, loss, performance) in file and console
         Creates tensorboard logs for each phase
         Saves source code in file
         Saves the best model and confusion matrix of this model
    - Runs train/test/devel phases
    - Calculates performance measures
    - Calculates confusion matrix
    - Saves models
    - Augments the data if needed (mixup)
     Args:
         log_root (str): Directory for logging
         experiment_name (str): Name of experiments for logging
         c_names (list[str]): Class names to calculate the confusion matrix
         metrics (list[callable]): List of performance measures based on the best results of which the model will be saved.
                                   The first measure (0) in the list will be used for this, the others provide
                                   additional information
         device (torch.device): Device where the model will be trained
         problem_type (ProblemType, optional): Problem type: for expression challenge - classification,
                                               for va challenge - regression. Defaults to ProblemType.CLASSIFICATION.
         group_predicts_fn (callable, optional): Function for grouping predicts, f.e. file-wise or windows-wise.
                                                 It can be used to calculate performance metrics on train/devel/test sets.
                                                 Defaults to None.
         source_code (str, optional): Source code and configuration for logging. Defaults to None.
         c_names_to_display (list[str], optional): Class names to visualize confuson matrix. Defaults to None.
    """

    def __init__(
        self,
        log_root: str,
        experiment_name: str,
        c_names: list[str],
        metrics: list[callable],
        device: torch.device,
        problem_type: ProblemType = ProblemType.CLASSIFICATION,
        group_predicts_fn: callable = None,
        source_code: str = None,
        c_names_to_display: list[str] = None,
    ) -> None:
        self.device = device

        self.model = None
        self.loss = None
        self.optimizer = None
        self.scheduler = None

        self.log_root = log_root
        self.exp_folder_name = experiment_name

        self.metrics = metrics
        self.c_names = c_names
        self.c_names_to_display = c_names_to_display
        self.problem_type = problem_type

        if source_code:
            os.makedirs(
                os.path.join(self.log_root, self.exp_folder_name, "logs"), exist_ok=True
            )
            with open(
                os.path.join(self.log_root, self.exp_folder_name, "logs", "source.log"),
                "w",
            ) as f:
                f.write(source_code)

        self.group_predicts_fn = group_predicts_fn

        self.logging_paths = None
        self.logger = None

    def create_loggers(self, fold_num: int = None) -> None:
        """Creates folders for logging experiments:
        - general logs (log_path)
        - models folder (model_path)
        - tensorboard logs (tb_log_path)

        Args:
            fold_num (int, optional): Used for cross-validation to specify fold number. Defaults to None.
        """
        fold_name = "" if fold_num is None else "fold_{0}".format(fold_num)
        self.logging_paths = {
            "log_path": os.path.join(self.log_root, self.exp_folder_name, "logs"),
            "model_path": os.path.join(
                self.log_root, self.exp_folder_name, "models", fold_name
            ),
            "tb_log_path": os.path.join(
                self.log_root, self.exp_folder_name, "tb", fold_name
            ),
        }

        for log_path in self.logging_paths:
            if log_path == "tb_log_path":
                continue

            os.makedirs(self.logging_paths[log_path], exist_ok=True)

        self.logger = create_logger(
            os.path.join(
                self.log_root,
                self.exp_folder_name,
                "logs",
                "{0}.log".format(fold_name if fold_name else "logs"),
            ),
            console_level=logging.NOTSET,
            file_level=logging.NOTSET,
        )

    def run(
        self,
        model: torch.nn.Module,
        loss: torch.nn.modules.loss,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        num_epochs: int,
        dataloaders: dict[torch.utils.data.dataloader.DataLoader],
        log_epochs: list[int] = [],
        fold_num: int = None,
        mixup_alpha: float = None,
        verbose: bool = True,
    ) -> None:
        """Iterates over epochs including the following steps:
        - Iterates over phases (train/devel/test phase):
            - Calls `iterate_model` function (as main loop for training/validation/testing)
            - Calculates performance measures (or metrics) using `calc_metrics` function
            - Performs logging
            - Compares performance with previous epochs on phase
            - Calculates confusion matrix
            - Saves better model and confusion matrix
            - Saves epoch/phase/loss/performance statistics in csv file

        Args:
            model (torch.nn.Module): Model instance
            loss (torch.nn.modules.loss): Loss function
            optimizer (torch.optim): Optimizer
            scheduler (torch.optim.lr_scheduler): Scheduler for dynamicly change LR
            num_epochs (int): Number of training epochs
            dataloaders (dict[torch.utils.data.dataloader.DataLoader]):
            log_epochs (list[int], optional): Exact epoch number for logging. Defaults to [].
            fold_num (int, optional): Used for cross-validation to specify fold number. Defaults to None.
            mixup_alpha (float, optional): Alpha value for mixup augmentation. Mixup is enabled if the value is set. Defaults to None.
            verbose (bool, optional): Detailed output including tqdm. Defaults to True.
        """
        phases = list(dataloaders.keys())
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.create_loggers(fold_num)
        d_global_stats = []

        summary = {}
        max_perf = {}
        for phase in phases:
            os.makedirs(
                os.path.join(self.logging_paths["tb_log_path"], phase), exist_ok=True
            )
            summary[phase] = SummaryWriter(
                os.path.join(self.logging_paths["tb_log_path"], phase)
            )

            main_metric_name = self.metrics[0].__name__
            max_perf[phase] = {
                "epoch": 0,
                "performance": {
                    main_metric_name: 0,
                },
            }

        self.logger.info(self.exp_folder_name)
        for epoch in range(1, num_epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, num_epochs))
            d_epoch_stats = {"epoch": epoch}

            for phase, dataloader in dataloaders.items():
                if "test" in phase and dataloader is None:
                    continue

                targets, predicts, sample_info, epoch_loss = self.iterate_model(
                    phase=phase,
                    dataloader=dataloader,
                    epoch=epoch,
                    mixup_alpha=mixup_alpha,
                    verbose=verbose,
                )
                self.logger.info(
                    "Epoch: {}. {}. Loss: {:.4f}, Performance:".format(
                        epoch, phase.capitalize(), epoch_loss
                    )
                )
                if self.problem_type == ProblemType.CLASSIFICATION:
                    performance = self.calc_metrics(
                        np.hstack(targets),
                        np.asarray(predicts).reshape(-1, len(self.c_names)),
                        verbose=verbose,
                    )
                else:
                    performance = self.calc_metrics(
                        np.stack(targets), np.stack(predicts), verbose=verbose
                    )

                d_epoch_stats["{}_loss".format(phase)] = epoch_loss
                summary[phase].add_scalar("loss", epoch_loss, epoch)

                epoch_score = performance[main_metric_name]
                for metric in performance:
                    summary[phase].add_scalar(metric, performance[metric], epoch)
                    d_epoch_stats["{}_{}".format(phase, metric)] = performance[metric]

                is_max_performance = (
                    (("test" in phase) or ("devel" in phase))
                    and (epoch_score > max_perf[phase]["performance"][main_metric_name])
                ) or (
                    (("test" in phase) or ("devel" in phase)) and (epoch in log_epochs)
                )

                if is_max_performance:
                    if epoch_score > max_perf[phase]["performance"][main_metric_name]:
                        max_perf[phase]["performance"] = performance
                        max_perf[phase]["epoch"] = epoch

                    if self.problem_type == ProblemType.CLASSIFICATION:
                        cm = conf_matrix(
                            np.hstack(targets),
                            np.asarray(predicts).reshape(-1, len(self.c_names)),
                            [i for i in range(len(self.c_names))],
                        )
                        res_name = "epoch_{0}_{1}_{2}".format(epoch, phase, epoch_score)
                        plot_conf_matrix(
                            cm,
                            labels=(
                                self.c_names_to_display
                                if self.c_names_to_display
                                else self.c_names
                            ),
                            xticks_rotation=45,
                            title="Confusion Matrix. {0}. UAR = {1:.3f}%".format(
                                phase, epoch_score * 100
                            ),
                            save_path=os.path.join(
                                self.logging_paths["model_path"],
                                "{0}.svg".format(res_name),
                            ),
                        )

                    model.cpu()
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss.state_dict(),
                        },
                        os.path.join(
                            self.logging_paths["model_path"],
                            "epoch_{0}.pth".format(epoch),
                        ),
                    )

                    model.to(self.device)

                if self.problem_type == ProblemType.CLASSIFICATION:
                    if os.path.exists(
                        os.path.join(
                            self.logging_paths["model_path"],
                            "epoch_{0}.pth".format(epoch),
                        )
                    ):
                        cm = conf_matrix(
                            np.hstack(targets),
                            np.asarray(predicts).reshape(-1, len(self.c_names)),
                            [i for i in range(len(self.c_names))],
                        )
                        res_name = "epoch_{0}_{1}_{2}".format(epoch, phase, epoch_score)
                        plot_conf_matrix(
                            cm,
                            labels=(
                                self.c_names_to_display
                                if self.c_names_to_display
                                else self.c_names
                            ),
                            xticks_rotation=45,
                            title="Confusion Matrix. {0}. UAR = {1:.3f}%".format(
                                phase, epoch_score * 100
                            ),
                            save_path=os.path.join(
                                self.logging_paths["model_path"],
                                "{0}.svg".format(res_name),
                            ),
                        )

            d_global_stats.append(d_epoch_stats)
            pd_global_stats = pd.DataFrame(d_global_stats)
            pd_global_stats.to_csv(
                os.path.join(
                    self.log_root,
                    self.exp_folder_name,
                    "logs",
                    (
                        "stats.csv"
                        if fold_num is None
                        else "fold_{0}.csv".format(fold_num)
                    ),
                ),
                sep=";",
                index=False,
            )

            self.logger.info("")

        for phase in phases[1:]:
            self.logger.info(phase.capitalize())
            self.logger.info(
                "Epoch: {}, Max performance:".format(max_perf[phase]["epoch"])
            )
            self.logger.info([metric for metric in max_perf[phase]["performance"]])
            self.logger.info(
                [
                    max_perf[phase]["performance"][metric]
                    for metric in max_perf[phase]["performance"]
                ]
            )
            self.logger.info("")

        for phase in phases:
            summary[phase].close()

        return model, max_perf

    def iterate_model(
        self,
        phase: str,
        dataloader: torch.utils.data.dataloader.DataLoader,
        epoch: int = None,
        mixup_alpha: float = None,
        verbose: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict], float]:
        """Main training/validation/testing loop:
        ! Note ! This loop needs to be changed if you change scheduler. Default scheduler is CosineAnnealingWarmRestarts
        - Applies softmax funstion on predicts if `problem_type` is ProblemType.CLASSIFICATION

        Args:
            phase (str): Name of phase: could be train, devel(valid), test
            dataloader (torch.utils.data.dataloader.DataLoader): Dataloader of phase
            epoch (int, optional): Epoch number. Defaults to None.
            mixup_alpha (float, optional): Alpha value for mixup augmentation. Mixup is enabled if the value is set. Defaults to None.
            verbose (bool, optional): Detailed output with tqdm. Defaults to True.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[dict], float]: targets,
                                                                          predicts,
                                                                          sample_info for grouping predicts/targets,
                                                                          epoch_loss
        """
        targets = []
        predicts = []
        sample_info = []

        if "train" in phase:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        has_labels = True
        iters = len(dataloader)

        # Iterate over data.
        for idx, data in enumerate(tqdm(dataloader, disable=not verbose)):
            inps, labs, s_info = data
            if isinstance(inps, list):
                inps = [d.to(self.device) for d in inps]
            else:
                inps = inps.to(self.device)

            if isinstance(labs, list):
                labs = [d.to(self.device) for d in labs]
            else:
                labs = labs.to(self.device)

            if self.problem_type == ProblemType.CLASSIFICATION:
                has_labels = torch.all(labs != -1)
            else:
                has_labels = True

            if (mixup_alpha) and ("train" in phase):
                inps, labs = self.mixup_data(inps, labs.flatten(), alpha=mixup_alpha)

            self.optimizer.zero_grad()

            # forward and backward
            preds = None
            with torch.set_grad_enabled("train" in phase):
                preds = self.model(inps)
                if self.loss:
                    if self.problem_type == ProblemType.CLASSIFICATION:
                        loss_value = self.loss(
                            preds.reshape(-1, len(self.c_names)), labs.flatten()
                        )
                    else:
                        loss_value = self.loss(
                            preds.reshape(-1, 2), labs.reshape(-1, 2)
                        )

                # backward + optimize only if in training phase
                if ("train" in phase) and has_labels and self.loss:
                    loss_value.backward()
                    self.optimizer.step()
                    if self.optimizer:
                        self.scheduler.step(epoch + idx / iters)

            # statistics
            if has_labels and self.loss:
                running_loss += loss_value.item() * dataloader.batch_size

            if isinstance(labs, list):
                labs = [d.cpu().numpy() for d in labs]
            else:
                labs = labs.cpu().numpy()

            targets.extend(labs)
            if self.problem_type == ProblemType.CLASSIFICATION:
                preds = F.softmax(preds, dim=-1)

            if isinstance(preds, list):
                preds = [d.cpu().detach().numpy() for d in preds]
            else:
                preds = preds.cpu().detach().numpy()

            predicts.extend(preds)
            sample_info.extend(s_info)

        epoch_loss = running_loss / iters if has_labels else 0

        if self.group_predicts_fn:
            targets, predicts, sample_info = self.group_predicts_fn(
                np.asarray(targets), np.asarray(predicts), sample_info
            )

        return targets, predicts, sample_info, epoch_loss

    def extract_features(
        self,
        phase: str,
        dataloader: torch.utils.data.dataloader.DataLoader,
        verbose: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict]]:
        """Loop for feature exctraction
        - Applies softmax funstion on predicts if `problem_type` is ProblemType.CLASSIFICATION

        Args:
            phase (str): Name of phase: could be train, devel(valid), test
            dataloader (torch.utils.data.dataloader.DataLoader): Dataloader of phase
            verbose (bool, optional): Detailed output with tqdm. Defaults to True.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict]]: targets,
                                                                                     predicts,
                                                                                     features,
                                                                                     sample_info
        """
        targets = []
        predicts = []
        features = []
        sample_info = []

        self.model.eval()

        # Iterate over data.
        for idx, data in enumerate(tqdm(dataloader, disable=not verbose)):
            inps, labs, s_info = data
            if isinstance(inps, list):
                inps = [d.to(self.device) for d in inps]
            else:
                inps = inps.to(self.device)

            if isinstance(labs, list):
                labs = [d.to(self.device) for d in labs]
            else:
                labs = labs.to(self.device)

            # forward and backward
            preds = None
            with torch.set_grad_enabled("train" in phase):
                preds, feats = self.model.get_features(inps)

            if isinstance(labs, list):
                labs = [d.cpu().numpy() for d in labs]
            else:
                labs = labs.cpu().numpy()

            targets.extend(labs)
            if self.problem_type == ProblemType.CLASSIFICATION:
                preds = F.softmax(preds, dim=-1)

            if isinstance(preds, list):
                preds = [d.cpu().detach().numpy() for d in preds]
            else:
                preds = preds.cpu().detach().numpy()

            predicts.extend(preds)

            feats = feats.cpu().detach().numpy()
            features.extend(feats)

            sample_info.extend(s_info)

        return targets, predicts, features, sample_info

    def calc_metrics(
        self,
        targets: list[np.ndarray],
        predicts: list[np.ndarray],
        verbose: bool = True,
    ) -> dict[float]:
        """Calculates each performance measure from `self.metrics`

        Args:
            targets (list[np.ndarray]): List of targets
            predicts (list[np.ndarray]): List of predicts
            verbose (bool, optional): Detailed output of each performance measure. Defaults to True.

        Returns:
            dict[float]: Return dictionary [performance.name] = value
        """

        performance = {}
        for metric in self.metrics:
            performance[metric.__name__] = metric(targets, predicts, average="macro")

        if verbose:
            self.logger.info([metric for metric in performance])
            if self.problem_type == ProblemType.CLASSIFICATION:
                self.logger.info(
                    [
                        "{0:.3f}".format(performance[metric] * 100)
                        for metric in performance
                    ]
                )
            else:
                self.logger.info(
                    ["{0:.3f}".format(performance[metric]) for metric in performance]
                )

        return performance

    def mixup_data(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies mixup augmentation on input data
        https://arxiv.org/abs/1710.09412

        Args:
            x (torch.Tensor): Tensor of features
            y (torch.Tensor): Tensor of labels
            alpha (float, optional): Alpha value. Defaults to 1.0.

        Raises:
            NotImplementedError: if ProblemType.REGRESSION

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Augmented features and labels
        """

        if self.problem_type == ProblemType.REGRESSION:
            raise NotImplementedError()

        lam = torch.FloatTensor([np.random.beta(alpha, alpha) if alpha > 0 else 1]).to(
            self.device
        )
        batch_size = x.size(0)
        y = F.one_hot(y, len(self.c_names))

        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y.argmax(1)
