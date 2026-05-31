import torch
from PIL import Image
from torchvision import transforms

from model import TrafficSignCNN
from config import *


CLASS_NAMES = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    # Добавь остальные классы
}


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


model = TrafficSignCNN(NUM_CLASSES)

model.load_state_dict(
    torch.load(MODEL_SAVE_PATH, map_location="cpu")
)

model.eval()


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    class_id = predicted.item()

    print(
        f"Prediction: {CLASS_NAMES[class_id]} | "
        f"Confidence: {confidence.item():.4f}"
    )


predict("test.jpg")