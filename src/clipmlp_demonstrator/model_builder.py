import sys
import os
import pickle
import sklearn

def get_requirements():
    return [
        "--index-url https://gitlab.com/api/v4/projects/70323435/packages/pypi/simple",
        "numpy==2.2.6",
        "scikit-learn==1.6.1",
        "scipy==1.15.3",
        "joblib==1.5.1",
        "threadpoolctl==3.6.0",
        "pillow==11.2.1",
        "torch==2.7.1",
        "filelock==3.18.0",
        "typing_extensions==4.14.0",
        "sympy==1.14.0",
        "networkx==3.5",
        "jinja2==3.1.6",
        "fsspec==2025.5.1",
        "setuptools==80.9.0",
        "mpmath==1.3.0",
        "MarkupSafe==3.0.2",
        "transformers==4.52.4",
        "huggingface_hub==0.32.4",
        "tokenizers==0.21.1",
        "packaging==25.0",
        "PyYAML==6.0.2",
        "regex==2024.11.6",
        "safetensors==0.5.3",
        "tqdm==4.67.1",
        "colorama==0.4.6",
        "requests==2.32.3",
        "charset_normalizer==3.4.2",
        "idna==3.10",
        "urllib3==2.4.0",
        "certifi==2025.4.26"
    ]

def get_network(model_name: str, task: str) -> nn.Module:
    if model_name == "clipmlp":   
        in_file = open("weights/mlp_wights.pkl", 'rb')
        model = pickle.load(in_file)
        in_file.close()
        return model
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Currently only 'clipmlp' is supported.")







##################




