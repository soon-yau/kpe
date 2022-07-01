from PIL import Image
import cv2
import torch
from torchvision import transforms as T
import numpy as np

class Segmentor:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 
                                    'deeplabv3_resnet101', pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to(self.device)
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __call__(self, image): # image is opencv
        input_image = Image.fromarray(image[:,:,::-1])
        input_image = input_image.convert("RGB")

        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]

        output_predictions = output.argmax(0)
        mask = (output_predictions.cpu().numpy()==15)*1
        mask = mask.astype(np.uint8)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        return masked_image, mask #[:,:,::-1]

