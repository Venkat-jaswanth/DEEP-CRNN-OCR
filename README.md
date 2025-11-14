## A Deep Convolutional Recurrent Neural Network (CRNN) for Optical Character Recognition

This project implements a Deep CRNN (Convolutional Recurrent Neural Network) for Optical Character Recognition (OCR) using PyTorch.
The model is trained only on isolated character images (A–Z, a–z, 0–9), but evaluated on unseen word images, demonstrating strong zero-shot generalization.

🔍 Overview

Traditional OCR systems struggle with noisy and stylized text such as CAPTCHA images.
This project solves that challenge using a CRNN model combining:

Component	Role
CNN (VGG-style)	Extracts visual features from text images
Bi-LSTM RNN	Learns character sequences and context
CTC Loss	Enables training without alignment

The trained model achieves 95%+ accuracy when recognizing word images it was never trained on — proving true character-learning rather than memorization.

✨ Features
- Trained only on single-character images
- Evaluated on unseen word images (zero-shot test)
- Handles font variations & Gaussian noise
- CTC decoding for variable-length transcription
- Visual output of predictions on test samples
- Fully implemented in PyTorch

🗂 Dataset Description

All datasets are synthetically generated using Python (Pillow).

Dataset	Usage	Variants	Content
captcha_images/	Training	Easy + Hard	Single characters
wordlist_captcha_images/	Testing	Easy + Hard	Full word images (80-word vocabulary, unseen during training)

Preprocessing

Resize → (200 × 80)

Convert to Tensor

Normalize pixel range to [-1, 1]



🧠 System Architecture (CRNN)
```
Input Image (200×80)
        ↓
Convolutional Neural Network (7 Conv layers, MaxPool, BatchNorm)
        ↓
Map-to-Sequence Layer
        ↓
Bi-Directional LSTM (2 layers, hidden size 256×2)
        ↓
Linear Classifier (63 classes: A-Z, a-z, 0-9, blank)
        ↓
CTC Loss / Decoding
        ↓
Recognized Text Output
```


⚙ Installation & Setup


```bash
git clone https://github.com/Venkat-jaswanth/DEEP-CRNN-OCR.git
cd DEEP-CRNN-OCR
pip install -r requirements.txt
```

Requirements include:
```
torch
torchvision
pandas
pillow
matplotlib
```

🏋️ Training

To train the CRNN on character dataset:

```
python DeepCRNN.ipynb   # or run in Jupyter Notebook
```

The trained model will be saved as:
```
DeepCRNN_model.pth
```

🧪 Testing & Evaluation

Testing is done using unseen wordlist images:
```
check_crnn_accuracy(model, test_dataloader)
```

Includes visualization of predictions (correct → green / incorrect → red)

📈 Results
Metric	Score
Word Test Accuracy	95%+
Loss Trend	Consistently decreasing across 15 epochs

Model correctly generalizes from characters → words ⭐


🚀 Applications

This CRNN can be extended into:

✔ CAPTCHA solvers
✔ Document digitization tools
✔ License plate recognition
✔ Automated data entry systems
✔ Assistive text-reading apps

🔮 Future Enhancements

Replace CNN backbone with ResNet/EfficientNet

Add Attention mechanism for improved decoding

Expand dataset to real-world OCR sources

Deploy via REST API / Web App / Mobile App
