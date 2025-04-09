import cv2
import torch
from torchvision import transforms
from model import NeuralNetwork

# clasificador preentrenado para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

# Definir transformaciones (igual que en el dataset de entrenamiento)
transform = transforms.Compose([
    transforms.ToPILImage(), # la imagen viene de OpenCV como array NumPy, y hay que convertirla antes de pasarla a Grayscale o Resize
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, im = cap.read()
    
    if ret == False:
        break
    
    # convertir la imagen a escala de grises (el detector lo necesita así)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # rectángulo alrededor de cada cara detectada
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Recortar la cara
        face = im[y:y+h, x:x+w]
        
        # Aplicar transformaciones
        face_tensor = transform(face).unsqueeze(0) 
        clases = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Predecir emoción:
        with torch.no_grad():
            output = model(face_tensor)
            predicted = torch.argmax(output, dim=1).item()
            emotion = clases[predicted]
            cv2.putText(im, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Face Detection', im)
    
    # tocar Esc para cerrar
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()