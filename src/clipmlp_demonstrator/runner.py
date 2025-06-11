import os
import time
from copy import deepcopy
import logging
import click
import mlflow
import pandas as pd
from PIL import Image
from PIL import ImageFile
from mlflow.models import infer_signature
from types import SimpleNamespace
import warnings
import numpy as np
from model_helpers.data_loader.image.data_frame import DataFrameLoader
from model_helpers.post_process.common import BasePostProcessor
from model_helpers.utils.codecs import pil_to_dataframe

from clipmlp_demonstrator.model_builder import get_requirements
from clipmlp_demonstrator.model_wrapper import ModelWrapper
from clipmlp_demonstrator.utils.uploader import upload_artifact
from clipmlp_demonstrator.runner_helper import ClipMlpRunner

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@click.command()
@click.option("--device", default='cuda:0', help="Device to use for training")
@click.option("--run_name", default='clipmlp-demonstrator', help="Name for the current run")
@click.option("--phase", multiple=True, default=['test'], help="Phases to execute (train/test)")
@click.option("--model_flag", default='clipmlp', help="Model architecture flag")
@click.option("--model_freeze", type=bool, default=False, help="Freeze base model weights")
@click.option("--min_vram", type=int, default=16000, help="Minimum VRAM requirement in MB")
@click.option("--load_id", type=int, default=16000, help="Minimum VRAM requirement in MB")

def run_method(device, phase, run_name, model_flag, model_freeze, min_vram, load_id):
    print("Current working directory:", os.getcwd())
    num_threads = os.cpu_count() // 2
    model = ClipMlpRunner(model_flag,device, load_id, num_threads, phase)
    # init the mlflow run for storing metrics and artifacts
    with mlflow.start_run(run_name=run_name) as run: 
        wrapper = ModelWrapper(
            model_class=model_flag,
            task=phase, 
            loader_class=DataFrameLoader,
            post_processors_class=BasePostProcessor   
        )
        image_path = './img_prova.png'
        image = Image.open(image_path).convert("RGB")
        prediction = np.argmax(model.predict(image)[0])
        mlflow.log_metric("generetor N", prediction)
        # infer signature
        upload_artifact( 
            image_path,
            artifact_path='clipmlp_demonstrator/weights/mlp_weights.pth',
            model_class=model_flag,
            model_freeze=model_freeze,
        )

if __name__ == "__main__":
    run_method()