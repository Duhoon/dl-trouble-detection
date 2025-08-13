import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import os.path as path
import os

class Classifier:
    def __init__(self):
        self.model = self.get_model()
        self.classes = os.listdir(path.join("dataset", "training", "resource"))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

    def classify_pic(self, image):
        transform = transforms.ToTensor()
        image = transform(image)
        image = image.unsqueeze(0)
        label = self.model(image)
        label = torch.argmax(label, dim=1).item()
        return self.idx_to_class[label]

    def get_model(self, ):
        model_path = path.join("model", "mobilenet_model_v2.pth")
        model = mobilenet_v2()
        num_ftrs = model.classifier[1].in_features  # 2048
        model.classifier[1] = torch.nn.Linear(num_ftrs, 6)
        state_dict= torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict=state_dict)
        print(model)
        return model
    