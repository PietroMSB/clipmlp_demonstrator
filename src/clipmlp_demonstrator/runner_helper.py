from abc import abstractmethod
import sys
import os
import pickle
import numpy as np
from pandas import DataFrame
import sklearn.neural_network
import sklearn.metrics
import sklearn.ensemble
import sklearn.svm
import sklearn.linear_model
import sklearn.neighbors
import json
import warnings
from transformers import CLIPModel, CLIPProcessor
from PIL import Image, ImageEnhance, ImageFilter, ImageFile
import torch

from clipmlp_demonstrator.model_builder import get_network
ImageFile.LOAD_TRUNCATED_IMAGES = True
import mlflow

class ClipMlpRunner:

    def __init__(self, model_flag,device, load_id, num_threads, task_type, run_directory: str = "runs/"):
        super().__init__()
        self.mlp = get_network(
            model_name=model_flag,
            task=task_type,
        )
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        # 
        self.run_directory = run_directory
        self.device = device
        self.load_id = load_id
        self.num_threads = num_threads
            
    @abstractmethod
    def predict(self, image: Image.Image | DataFrame) -> np.array:      
        #extract features with clip
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        #normalize features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        #convert features to numpy array
        image_features_np = np.reshape(image_features.cpu().numpy()[0], (1,-1))
        #predict
        return self.mlp.predict(torch.from_numpy(image_features_np)).cpu().numpy()


