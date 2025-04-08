import torch.nn as nn

class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding='same') 
		"""
		-in_channels = cantidad de canales que tiene la imagen de entrada. 1 canal = escala de grises.

		-out_channels define cuántos filtros o "detectores de patrones" vas a usar en esa capa. 
		En las primeras capas suele usarse un número bajo (como 6 o 16) y va subiendo a medida que se profundiza.

		-Un kernel es como una mini ventana que se desliza por toda la imagen. 
		Si ponés kernel_size=3, significa que esa ventanita mide 3x3 píxeles.
		Lo que hace es detectar patrones en esas zonas: bordes, curvas, texturas...

		-padding=1 agrega 1 pixel de "borde" alrededor de la imagen (arriba, abajo, izquierda, derecha). 
		Esto se hace para que la imagen no se achique tanto con cada convolución.
		padding='same' → agrega el padding justo para que la salida tenga el mismo tamaño que la entrada.
		"""
		self.bn1 = nn.BatchNorm2d(8)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(2,2) # reduce a la mitad el alto y ancho

		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same')
		self.bn2 = nn.BatchNorm2d(16)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(2,2)

		self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
		self.bn3 = nn.BatchNorm2d(32)
		self.relu3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(2,2)

		self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()
		self.pool4 = nn.MaxPool2d(2,2)

		self.dropout = nn.Dropout(0.5)
		"""Dropout apaga el 50% de las neuronas aleatoriamente durante cada paso de entrenamiento.
		Sirve para que la red no dependa tanto de neuronas específicas, ayudando a evitar el sobreentrenamiento.
		0.5 es un valor común, pero también podés probar con 0.3 o 0.25."""
		self.flatten = nn.Flatten()

		# Capa totalmente conectada 1
		self.fc1 = nn.Linear(64 * 3 * 3, 512)  # Ajusta la cantidad de nodos según las dimensiones
		# Capa totalmente conectada 2
		self.fc2 = nn.Linear(512, 256)  # Nueva capa
		# Capa de salida
		self.fc3 = nn.Linear(256, 7)

		"""
		¿Por qué 64 * 3 * 3? 
		Imagen original: 48x48 (en FER2013).
		Después de 4 pools, cada uno divide ancho y alto por 2 → 48 → 24 → 12 → 6 → 3
 		Última capa tiene 64 canales → entonces el vector final tiene 64 * 3 * 3 elementos.

		¿Por qué 512?
		Acá podés elegir casi cualquier número, pero:
		Tiene que ser menor que la cantidad de entradas (o sea, menor que 64*3*3 = 576)
		Y suficientemente grande como para que la red tenga capacidad de aprender cosas útiles.
		512 es un número bastante estándar y balanceado, ni muy chico (que se quede corta la red), ni muy grande (que se sobreentrene o pese mucho).
		"""
        
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.pool1(x)
		
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		x = self.pool2(x)
		
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu3(x)
		x = self.pool3(x)
		
		x = self.conv4(x)  # Nueva capa convolucional
		x = self.bn4(x)
		x = self.relu4(x)
		x = self.pool4(x)

		x = self.dropout(x)
		x = self.flatten(x)
		
		x = self.fc1(x)
		x = self.fc2(x)  # Nueva capa completamente conectada
		x = self.fc3(x)

		return x