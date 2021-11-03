"""
Predictor interfaces for the Deep Learning challenge.
"""

import os
from typing import List

import torch
import torchvision.transforms as T
import numpy as np

from deep_equation.model import Model
from deep_equation.transformations import image_transform

class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """

    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions


default_model_path = '../../resources/models/modelo_desafio_mod_lenet5_norm.pth'

class StudentModel(BaseNet):
    def load_model(self, model_path: str = default_model_path, device='cpu'):
        """
        Load the student's trained model.
        """
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, model_path)
        model = Model()
        model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
        return model
    
    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ):
        model = self.load_model(device=device)
        model.eval()

        processed_images_a = torch.stack([self.preprocess(img) for img in images_a]).to(device)
        processed_images_b = torch.stack([self.preprocess(img) for img in images_b]).to(device)
        operator_indexes = self.operator_indexes(operators).to(device)
        
        with torch.no_grad():
            outputs = model(processed_images_a, processed_images_b, operator_indexes)
            outputs = outputs.argmax(1)
            predictions = [float(model.classes[outputs[i]]) for i in range(len(outputs))]
            return predictions

    def preprocess(self, image):
        image = image_transform()(image)
        return T.functional.adjust_brightness(image, 10)

    def operator_indexes(self, operators):
        return torch.tensor([[self.operator_index(operator)] for operator in operators])

    def operator_index(self, operator):
        if operator == '+':
            return 0
        elif operator == '-':
            return 1
        elif operator == '*':
            return 2
        elif operator == '/':
            return 3
