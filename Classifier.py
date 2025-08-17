import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, densenet121
import os.path as path
import os
import pandas as pd

class Classifier:
    def __init__(self):
        self.model = {
            "densenet": self.get_densenet_model(),
            "mobilenet": self.get_mobilenet_model(),
        }
        self.classes = ["건선", "아토피", "여드름", "정상", "주사", "지루"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

    def classify_pic(self, image, model_name):
        if not model_name in self.model.keys():
            raise "모델이 주어지지 않았습니다."
        
        transform = transforms.ToTensor()
        image = transform(image)
        image = image.unsqueeze(0)
        
        with torch.no_grad():
            pred = self.model[model_name](image)
            prob = torch.nn.functional.softmax(pred.flatten(), dim=0)
            prob_sorted, prob_sorted_idx = prob.sort(dim=0, descending=True)

            prob_sorted_labels = [ self.idx_to_class[idx] for idx in prob_sorted_idx.tolist()]

            probs = pd.DataFrame({"질환": prob_sorted_labels, "확률": prob_sorted.tolist()})
            label = self.idx_to_class[torch.argmax(pred, dim=1).item()]
        
        return label, probs

    def get_mobilenet_model(self):
        model_path = path.join("model", "mobilenet_model_v2.pth")
        model = mobilenet_v2()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, 6)
        state_dict= torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict=state_dict)
        model.eval()
        return model
    
    def get_densenet_model(self):
        model_path = path.join("model", "densenet_model.pth")
        model = densenet121()
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, 6)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict=state_dict)
        model.eval()
        return model