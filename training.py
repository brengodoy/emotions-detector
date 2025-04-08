from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from model import NeuralNetwork
import os
import torch
import torch.nn as nn
from tqdm import tqdm

data_dir = "dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Grayscale(),       # Convierte la imagen a escala de grises (1 canal, como vos querés)
    transforms.Resize((48, 48)),  # Cambia el tamaño a 48x48 píxeles (que es lo que usa FER2013)
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()         # Convierte la imagen en un tensor, o sea en una estructura que PyTorch puede usar
])

# Cargar datasets
train_dataset = ImageFolder(train_dir, transform=transform) # aplica transformaciones a las imagenes, los convierte en tensores
test_dataset = ImageFolder(test_dir, transform=transform)

BATCH_SIZE = 128

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)

LEARNING_RATE = 0.001
EPOCHS = 50

loss_fn = nn.CrossEntropyLoss()  # funcion ideal cuando tenemos varias clases, en este caso hay 7 clases.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Aprende más rápido que SDG. Se adapta mejor al "ritmo" del aprendizaje (ajusta automáticamente el learning rate para cada parámetro)

def train_loop(dataloader, model, loss_fn, optimizer):
    train_size = len(dataloader.dataset)
    batch_quantity = len(dataloader)
    
    model.train()
    
    train_loss, accuracy = 0, 0
    
    for batch_number, (X, y) in enumerate(tqdm(dataloader, desc="📚 Entrenando", leave=True)):
        X, y = X.to(device), y.to(device)
        
        logits = model(X)
        
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        accuracy += (logits.argmax(1) == y).type(torch.float).sum().item()
            
    train_loss /= batch_quantity
    accuracy /= train_size
    
    print("Training: Average loss: " + str(train_loss) + ". Accuracy: " + str(100*accuracy) + "%")
        
def validation_loop(dataloader, model, loss_fn):
    val_size = len(dataloader.dataset)
    batch_quantity = len(dataloader)
    
    model.eval()
    
    validation_loss, accuracy = 0, 0
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validando", leave=False):
            X, y = X.to(device), y.to(device)
            
            logits = model(X)
            
            validation_loss += loss_fn(logits, y).item()
            accuracy += (logits.argmax(1) == y).type(torch.float).sum().item()
    
    validation_loss /= batch_quantity
    accuracy /= val_size
    
    print(f"Validación: Pérdida promedio: {validation_loss:.2f}. Exactitud: {accuracy*100:.2f}%")
    return validation_loss, accuracy
        
for i in range(EPOCHS):
    print(f"\n🌸 Época {i+1}/{EPOCHS}")
    train_loop(train_loader, model, loss_fn, optimizer)
    validation_loop(test_loader, model, loss_fn)
    
torch.save(model.state_dict(), 'model.pth')
print("Entrenamiento terminado!")