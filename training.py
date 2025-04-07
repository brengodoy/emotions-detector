from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from model import NeuralNetwork
import os
import torch
import torch.nn as nn

data_dir = "dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Grayscale(),       # Convierte la imagen a escala de grises (1 canal, como vos querés)
    transforms.Resize((48, 48)),  # Cambia el tamaño a 48x48 píxeles (que es lo que usa FER2013)
    transforms.ToTensor()         # Convierte la imagen en un tensor, o sea en una estructura que PyTorch puede usar
])

# Cargar datasets
train_dataset = ImageFolder(train_dir, transform=transform) # aplica transformaciones a las imagenes, los convierte en tensores
test_dataset = ImageFolder(test_dir, transform=transform)

BATCH_SIZE = 64

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)

LEARNING_RATE = 0.1
EPOCHS = 20

loss_fn = nn.CrossEntropyLoss()  # funcion ideal cuando tenemos varias clases, en este caso hay 7 clases.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Aprende más rápido que SDG. Se adapta mejor al "ritmo" del aprendizaje (ajusta automáticamente el learning rate para cada parámetro)

def train_loop():
    pass