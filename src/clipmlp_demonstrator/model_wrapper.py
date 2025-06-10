import logging
from typing import Type
from PIL.Image import Image
from pandas import DataFrame
from tqdm import tqdm
import torch
import os 
import pickle
import numpy as np
from model_helpers.data_loader.common import AbstractDataLoader
from model_helpers.post_process.common import AbstractPostProcessor
from model_helpers.utils.codecs import dataframe_to_pil
from model_helpers.utils.device import get_device
from model_helpers.wrapper import ModelWrapper as BaseWrapper
from transformers import CLIPModel, CLIPProcessor

from clipmlp_demonstrator.model_builder import get_network, MLP

logger = logging.getLogger(__name__)

class ModelWrapper(BaseWrapper):
    def __init__(self,
                 model_class: str,
                 task: str, 
                 loader_class: Type[AbstractDataLoader],
                 post_processors_class: Type[AbstractPostProcessor],
                 **kwargs
                 ):
        super().__init__(
            loader_class=loader_class,
            post_processors_class=post_processors_class,
            **kwargs
        )
        self._model_class = model_class
        self.task = task        
        self.mlp = None
        self.clip_model = None
        self.clip_processor = None
        self.device = None
        # DO NOT initialize the model in __init__ method, the model should be initialized in load_context method (https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel.load_context)

    def load_context(self, context):
        super().load_context(context)
        logger.info(f"Calling load_context with context: {context}")
        logger.info(f"artifacts: {context.artifacts}")
        artifact_path = context.artifacts["mlp"]
        self.load_model(artifact_path)

    def load_model(self, artifact_path):
        self.device = get_device(None)
        logger.info(f"device {self.device}")
        self.mlp = get_network(
            model_name=self._model_class,
            task=self.task
        )
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        logger.info(f"Loading model from: {artifact_path}")
        self.mlp = get_network("clipmlp", "task")
        self.mlp.load_state_dict(torch.load(artifact_path))
        
    def preprocess_image(self, device):
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        return self.clip_processor

    def run_prediction(self, image: Image | DataFrame) -> np.array:
        if not isinstance(image, Image):
             image = dataframe_to_pil(image, "RGB")           
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
        #process image with CLIP
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        #normalize features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        #convert features to numpy array
        image_features_np = np.reshape(image_features.cpu().numpy()[0], (1,-1))
        #predict
        return self.mlp.predict(torch.from_numpy(image_features_np))
