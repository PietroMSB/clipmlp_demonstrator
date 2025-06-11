import sys
import os
import pickle
import sklearn
import torch

#hyperparameters for MLP
mlp_hidden_layers = (512, 256)
mlp_activation = torch.nn.ReLU
mlp_optimizer = torch.optim.Adam
mlp_ilr = 0.001
mlp_batch_size = 1000
mlp_epochs = 1000
mlp_tolerance = 10

#MLP class
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation, save_path="clipmlp_demonstrator/weights/mlp_weights.pth"):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_size, h))
            layers.append(activation())
            in_size = h
        layers.append(torch.nn.Linear(in_size, output_size))
        self.network = torch.nn.Sequential(*layers)
        self.save_path = save_path

    def forward(self, x):
        return self.network(x)
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return torch.nn.functional.softmax(self.forward(x))
        
    def train_step(self, data, target, optimizer, criterion):
        #forward
        output = self.forward(data)
        loss = criterion(output, target)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def train_model(self, X_tr, Y_tr, X_va, Y_va, optimizer_fn, criterion, ilr, num_epochs, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        optimizer = optimizer_fn(self.parameters(), lr = ilr)
        for epoch in range(num_epochs):
            self.train()
            train_loss =  self.train_step(X_tr, Y_tr, optimizer, criterion)
            #validation step
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                output = self.forward(X_va)
                val_loss = criterion(output, Y_va).item()
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            #early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), self.save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        #restore best weights
        self.load_state_dict(torch.load(self.save_path))
        print("Restored best model weights.")
    

def get_requirements():
    return [
        "--index-url https://gitlab.com/api/v4/projects/70323435/packages/pypi/simple",
        "model_helpers==0.1.0",
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
        "packaging==24.2",
        "PyYAML==6.0.2",
        "regex==2024.11.6",
        "safetensors==0.5.3",
        "tqdm==4.67.1",
        "colorama==0.4.6",
        "requests==2.32.3",
        "charset_normalizer==3.4.2",
        "idna==3.10",
        "urllib3==2.4.0",
        "certifi==2025.4.26",
        "mlflow==2.22.0",
        "mlflow-skinny==2.22.0"
    ]

def get_network(model_name: str, task: str):
    if model_name == "clipmlp":   
        return MLP(768, mlp_hidden_layers, 10, mlp_activation)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Currently only 'clipmlp' is supported.")







##################




