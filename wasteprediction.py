import io
import os
import shutil
import sys
import tarfile
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from zipfile import ZipFile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

data_dir = "Dataset"

classes = os.listdir(data_dir)

CHUNK_SIZE = 40960

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = ImageFolder(data_dir, transform = transformations)

random_seed = 42
torch.manual_seed(random_seed)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.05
TEST_RATIO = 0.15

train_ds, val_ds, test_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO])

from torch.utils.data.dataloader import DataLoader

batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50()
        self.network.load_state_dict(torch.load('model/resnet50-0676ba61.pth'))
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

model = ResNet()


state_dict = torch.load('model/garbage_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

app = Flask(__name__)

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img_bytes = file.read()
        prediction = get_prediction(img_bytes)
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


