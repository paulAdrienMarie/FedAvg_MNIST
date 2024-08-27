import platform
import onnx
from onnxruntime.training import artifacts
import os
import torch
import io
assert list(platform.python_version_tuple())[:-1] == ["3", "9"]

class MNISTNET(torch.nn.Module):
    """
    
    Class for the MNIST model
    
    Attributs:
    fc1 -- First hidden layer
    relu -- Relu Activation
    fc2 -- Second hidden layer
    """
    
    def __init__(self, input_size, hidden_size, num_classes):
        """
        
        Initializes a new instance of the Model class
        
        input_size -- size of the input image
        hidden_size -- size of the hidden layer
        num_classes -- number of classes in the model
        
        """
        super(MNISTNET, self).__init__()
        
        self.fc1 = torch.nn.Linear(input_size,hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        
        
    def forward(self, model_input):
        """
        
        Runs the model with given inputs
        
        model_input -- inputs of the model
        
        """
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Artifacts: 
    """
    
    Class to handle the generation of the training artifacts
    
    Attributs:
    model_path -- Path to the onnx model
    artifacts_path -- Path where to save the artifacts
    
    """
    
    def __init__(self, model_path, artifacts_path):
        """
        
        Initializes a new instance of the Artifacts class
        
        model_path -- path to the onnx model
        artifacts_path -- path where to save the training artifacts
        
        """
        
        self.model_path = model_path
        self.artifacts_path = artifacts_path
        self.model = None
          
    def export_model(self):
        """Exports the MNIST model in onnx format"""
        
        device = "cpu"
        batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
        self.model = MNISTNET(input_size, hidden_size, output_size).to(device)
        
        model_inputs = (torch.randn(batch_size,input_size,device=device),)
        
        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        
        f = io.BytesIO()
        torch.onnx.export(
            self.model,
            model_inputs,
            f,
            input_names=input_names,
            output_names=output_names,
            opset_version=14,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamic_axes=dynamic_axes,
            export_params=True,
            keep_initializers_as_inputs=False
        )
        
        
        onnx_model = onnx.load_model_from_string(f.getvalue())
        
        onnx.save_model(onnx_model,self.model_path)
        
    def gen_artifacts(self):
        """Generates the training artifacts"""

        onnx_model = onnx.load_model(self.model_path)
        
        requires_grad = [param.name for param in onnx_model.graph.initializer]
        frozen_params = [param.name for param in onnx_model.graph.initializer if param.name not in requires_grad]
        
        output_names = ["output"]
        
        artifacts.generate_artifacts(
            onnx_model,
            optimizer=artifacts.OptimType.AdamW,
            loss=artifacts.LossType.CrossEntropyLoss,
            requires_grad=requires_grad,
            frozen_params=frozen_params,
            additional_output_names=output_names,
            artifact_directory=os.path.join(os.path.dirname(__file__), "artifacts")
        )
        
    def __call__(self):
        """Generates training artifacts"""
        self.export_model()
        self.gen_artifacts()

if __name__=="__main__":
    
    artifacts_path = os.path.join(os.path.dirname(__file__), "artifacts")
    model_path = os.path.join(os.path.dirname(__file__), "onnx/inference.onnx")
    
    obj = Artifacts(
        model_path=model_path,
        artifacts_path=artifacts_path,
    )
    obj()