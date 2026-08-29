---
url: https://cdn.jsdelivr.net/gh/shivammehta25/Matcha-TTS@main/matcha/train.py
title: 
retrieved_at: 2026-08-22T20:08:03Z
api_version: v2
---

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from matcha import utils

rootutils.setup\_root(\_\_file\_\_, indicator=".project-root", pythonpath=True)
\# ------------------------------------------------------------------------------------ #
\# the setup\_root above is equivalent to:
\# - adding project root dir to PYTHONPATH
\# (so you don't need to force user to install project as a package)
\# (necessary before importing any local modules e.g. \`from src import utils\`)
\# - setting up PROJECT\_ROOT environment variable
\# (which is used as a base for paths in "configs/paths/default.yaml")
\# (this way all filepaths are the same no matter where you run the code)
\# - loading environment variables from ".env" in root dir
#
\# you can remove it if you:
\# 1\. either install project as a package or move entry files to project root dir
\# 2\. set \`root\_dir\` to "." in "configs/paths/default.yaml"
#
\# more info: https://github.com/ashleve/rootutils
\# ------------------------------------------------------------------------------------ #

log = utils.get\_pylogger(\_\_name\_\_)

@utils.task\_wrapper
def train(cfg: DictConfig) -> Tuple\[Dict\[str, Any\], Dict\[str, Any\]\]:
 """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
 training.

 This method is wrapped in optional @task\_wrapper decorator, that controls the behavior during
 failure. Useful for multiruns, saving info about the crash, etc.

 :param cfg: A DictConfig configuration composed by Hydra.
 :return: A tuple with metrics and dict with all instantiated objects.
 """
 # set seed for random number generators in pytorch, numpy and python.random
 if cfg.get("seed"):
 L.seed\_everything(cfg.seed, workers=True)

 log.info(f"Instantiating datamodule <{cfg.data.\_target\_}>") # pylint: disable=protected-access
 datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

 log.info(f"Instantiating model <{cfg.model.\_target\_}>") # pylint: disable=protected-access
 model: LightningModule = hydra.utils.instantiate(cfg.model)

 log.info("Instantiating callbacks...")
 callbacks: List\[Callback\] = utils.instantiate\_callbacks(cfg.get("callbacks"))

 log.info("Instantiating loggers...")
 logger: List\[Logger\] = utils.instantiate\_loggers(cfg.get("logger"))

 log.info(f"Instantiating trainer <{cfg.trainer.\_target\_}>") # pylint: disable=protected-access
 trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

 object\_dict = {
 "cfg": cfg,
 "datamodule": datamodule,
 "model": model,
 "callbacks": callbacks,
 "logger": logger,
 "trainer": trainer,
 }

 if logger:
 log.info("Logging hyperparameters!")
 utils.log\_hyperparameters(object\_dict)

 if cfg.get("train"):
 log.info("Starting training!")
 trainer.fit(model=model, datamodule=datamodule, ckpt\_path=cfg.get("ckpt\_path"))

 train\_metrics = trainer.callback\_metrics

 if cfg.get("test"):
 log.info("Starting testing!")
 ckpt\_path = trainer.checkpoint\_callback.best\_model\_path
 if ckpt\_path == "":
 log.warning("Best ckpt not found! Using current weights for testing...")
 ckpt\_path = None
 trainer.test(model=model, datamodule=datamodule, ckpt\_path=ckpt\_path)
 log.info(f"Best ckpt path: {ckpt\_path}")

 test\_metrics = trainer.callback\_metrics

 # merge train and test metrics
 metric\_dict = {\*\*train\_metrics, \*\*test\_metrics}

 return metric\_dict, object\_dict

@hydra.main(version\_base="1.3", config\_path="../configs", config\_name="train.yaml")
def main(cfg: DictConfig) -> Optional\[float\]:
 """Main entry point for training.

 :param cfg: DictConfig configuration composed by Hydra.
 :return: Optional\[float\] with optimized metric value.
 """
 # apply extra utilities
 # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
 utils.extras(cfg)

 # train the model
 metric\_dict, \_ = train(cfg)

 # safely retrieve metric value for hydra-based hyperparameter optimization
 metric\_value = utils.get\_metric\_value(metric\_dict=metric\_dict, metric\_name=cfg.get("optimized\_metric"))

 # return optimized metric
 return metric\_value

if \_\_name\_\_ == "\_\_main\_\_":
 main() # pylint: disable=no-value-for-parameter