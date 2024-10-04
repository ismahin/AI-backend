import torch
from torchvision import transforms

class DiseaseModel:
    def __init__(self):
        # Load your trained model here
        # For demonstration, we're using a pre-trained ResNet18
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.model.eval()

        # Define the class names
        self.class_names = ['Apple Scab', 'Apple Black Rot', 'Grape Black Measles', 'Tomato Early Blight', 'Healthy']

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image):
        # Preprocess the image
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch

        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_class = torch.max(probabilities, dim=0)
        disease_name = self.class_names[top_class % len(self.class_names)]  # Adjust index if necessary

        return disease_name
