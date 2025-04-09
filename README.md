# 🎭 Emotion Detector with CNN

This project implements a Convolutional Neural Network (CNN) for real-time facial emotion recognition using the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013). The model is capable of detecting 7 emotions from facial expressions: **angry**, **disgust**, **fear**, **happy**, **sad**, **surprise**, and **neutral**.

---

## 📌 Project Structure

- `training.py`: Script to train the CNN model on the FER2013 dataset.
- `model.py`: Definition of the neural network architecture using PyTorch.
- `app.py`: Real-time emotion recognition using OpenCV and a webcam.
- `model.pth`: Trained model weights (exported after training).

---

## 🧠 Model Architecture

The architecture consists of 4 convolutional blocks (`Conv2D + BatchNorm + ReLU + MaxPooling`), followed by a fully connected head:

```python
self.conv1 → self.conv2 → self.conv3 → self.conv4 → flatten → fc1 → fc2
```

Dropout is used to reduce overfitting. The final layer outputs a 7-class prediction using softmax.

---

## 📊 Training

- **Dataset**: FER2013 (48x48 grayscale facial images)
- **Accuracy achieved**: ~56% on validation
- **Loss function**: `CrossEntropyLoss`
- **Optimizer**: `Adam`, learning rate = 0.001
- **Epochs**: 50
- **Data Augmentation**:
  - Random horizontal flip
  - Random rotation

Training was performed in **Google Colab** using GPU acceleration. The final model was exported to a `.pth` file and used locally for inference.

---

## 🔍 Inference Pipeline

1. Open webcam feed using OpenCV.
2. Detect face using Haar Cascades.
3. Preprocess the detected face:
   - Convert to grayscale
   - Resize to 48x48
   - Convert to PyTorch tensor
4. Pass the image through the trained model.
5. Display the predicted emotion on screen using `cv2.putText`.

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Main libraries used**:

- `torch`
- `torchvision`
- `opencv-python`
- `imutils`

---

## 🚀 How to Run

- **To train the model**:

```bash
python training.py
```

- **To run the real-time emotion detector**:

```bash
python app.py
```

> Make sure `model.pth` is in the same directory as `app.py`.

---

## 📝 Notes

- The model was trained on grayscale images resized to 48x48 pixels, so input preprocessing must match.
- You can further improve performance by:
  - Using a deeper CNN (e.g., ResNet)
  - Applying more aggressive data augmentation
  - Training on a larger or more balanced dataset
- Emotion detection is approximate and works best under good lighting conditions.

---

## 📸 Preview

![image](https://github.com/user-attachments/assets/848a45f1-21ae-46be-8ddc-0fad67101147)

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE.md) file for details.
