import platform
import onnx
import torch
from Artifacts import Artifacts
import json
import os
from onnxruntime.training.api import CheckpointState, Module, Optimizer
import numpy as np
from torchvision import datasets, transforms

assert list(platform.python_version_tuple())[:-1] == ["3", "9"]

class Train:
    """
    
    Class to train the MNIST model
    
    Attributs:
    
    path_to_training -- path to the file of the training model
    path_to_eval -- path to the file of the eval model
    path_to_optimizer -- path to the file of the optimizer model
    path_to_checkpoint -- path to the checkpoint file 
    path_to_model -- path to the file of the model to train
    path_to_config -- path to config files

    """
    
    def __init__(
        self,
        path_to_training,
        path_to_eval,
        path_to_optimizer,
        path_to_checkpoint,
        path_to_model,
        path_to_config
    ):
        """
        
        Initializes a new instance of the Train class
        
        Arguments:
        
        path_to_training -- path to the file of the training model
        path_to_eval -- path to the file of the eval model
        path_to_optimizer -- path to the file of the optimizer model
        path_to_checkpoint -- path to the checkpoint file 
        path_to_model -- path to the file of the model to train
        path_to_config -- path to config files

        """
        
        self.path_to_training = path_to_training
        self.path_to_eval = path_to_eval
        self.path_to_optimizer = path_to_optimizer
        self.path_to_checkpoint = path_to_checkpoint
        self.path_to_model = path_to_model
        self.path_to_config = path_to_config
        self.model = None
    
    def load_train_images(self):
        """Loads training images"""
        
        batch_size = 64
        train_kwargs = {'batch_size': batch_size}
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST('./', train=True, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
        return train_loader

        
    def load_model(self):
        """Loads the model"""
        print("Loading the model")
        self.model = onnx.load_model("./onnx/inference.onnx")
        
    def loadJson(self, path):
        """Loads json file as dictionnary"""
        with open(path) as f:
            return json.loads(f.read())
        
    def preprocess_images_to_numpy(self, images):
        """
        
        Preprocess images and returns them as numpy arrays
        
        images -- the images to preprocess
        
        """
        
        pre = self.loadJson(self.path_to_config)

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[pre["mean"]], std=[pre["std"]])
        ])

        preprocessed_images = [transform(img) for img in images]
        images_tensor = torch.stack(preprocessed_images)
        return images_tensor.view(images_tensor.size(0), -1).numpy()

    def load_training_modules(self):
        """Loads the training modules"""
        
        print("Loading training modules")
        state = CheckpointState.load_checkpoint(self.path_to_checkpoint)
        module = Module(
            self.path_to_training,
            state,
            self.path_to_eval,
            device="cpu"
        )
        optimizer = Optimizer(self.path_to_optimizer, module)
        return module, optimizer, state
    
    def generate_target_logits(self, y_true):
        """Generates target logits"""
        return np.int64(y_true)
        
    def train(self, train_loader,epoch):
        """
        
        Run the training loop
        
        Arguments:
        train_loader -- The set of training images
        epoch -- The current epoch 
        
        """
        module, optimizer, state = self.load_training_modules()
        
        module.train()
        losses = []
        for _, (data, target) in enumerate(train_loader):
            forward_inputs = [data.reshape(len(data),784).numpy(),target.numpy().astype(np.int64)]
            train_loss, _ = module(*forward_inputs)
            print(f"LOSS : {train_loss}")
            optimizer.step()
            module.lazy_reset_grad()
            losses.append(train_loss)

        print(f'Epoch: {epoch+1} - Train Loss: {sum(losses)/len(losses):.4f}')
        
        CheckpointState.save_checkpoint(state, self.path_to_checkpoint)
        module.export_model_for_inferencing("./onnx/mnist_trained.onnx", ["output"])
        
    def __call__(self,epoch):
        train_loader = self.load_train_images()
        self.train(train_loader,epoch)

class Test:
    """
    
    Class to test the MNIST model
    
    Attributs:
    
    path_to_training -- path to the file of the training model
    path_to_eval -- path to the file of the eval model
    path_to_optimizer -- path to the file of the optimizer model
    path_to_checkpoint -- path to the checkpoint file 
    path_to_model -- path to the file of the model to train
    path_to_config -- path to config files

    """
    
    def __init__(
        self,
        path_to_training,
        path_to_eval,
        path_to_optimizer,
        path_to_checkpoint,
        path_to_model,
        path_to_config
    ):
        """
        
        Initializes a new instance of the Train class
        
        Arguments:
        
        path_to_training -- path to the file of the training model
        path_to_eval -- path to the file of the eval model
        path_to_optimizer -- path to the file of the optimizer model
        path_to_checkpoint -- path to the checkpoint file 
        path_to_model -- path to the file of the model to train
        path_to_config -- path to config files

        """
        
        self.path_to_training = path_to_training
        self.path_to_eval = path_to_eval
        self.path_to_optimizer = path_to_optimizer
        self.path_to_checkpoint = path_to_checkpoint
        self.path_to_model = path_to_model
        self.path_to_config = path_to_config
        self.model = None
    
    def loadJson(self, path):
        """Loads json file as dictionnary"""
        with open(path) as f:
            return json.loads(f.read())
   
    def load_test_images(self):
        """Loads test images"""
        
        pre = self.loadJson(self.path_to_config)
        
        test_batch_size = 1000
        test_kwargs = {'batch_size': test_batch_size}
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[pre["mean"]], std=[pre["std"]])
        ])
        
        dataset2 = datasets.MNIST('./', train=False,
                    transform=transform)
        
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        
        return test_loader
    
    def argmax(self, logits):
        """
        
        Return argmax of the given logits
        
        logits -- The raw outputs of the model
        
        """
        return np.argmax(logits, axis=-1) 
    
    
    def test(self,test_loader,epoch):
        """
        
        Run the testing loop
        
        Arguments:
        test_loader -- The set of testing images
        epoch -- The current epoch 
        
        """
        
        state = CheckpointState.load_checkpoint("./artifacts/checkpoint")

        module = Module(
            "./artifacts/training_model.onnx",
            state,
            "./artifacts/eval_model.onnx",
            device="cpu"
        )
    
        module.eval()
        losses = []
        import evaluate
        metric = evaluate.load('accuracy')

        for _, (data, target) in enumerate(test_loader):
            forward_inputs = [data.reshape(len(data),784).numpy(),target.numpy().astype(np.int64)]
            test_loss, logits = module(*forward_inputs)
            metric.add_batch(references=target, predictions=self.argmax(logits))
            losses.append(test_loss)

        metrics = metric.compute()
        print(f'Epoch: {epoch+1} - Test Loss: {sum(losses)/len(losses):.4f}, Accuracy : {metrics["accuracy"]:.2f}')
        
    def __call__(self,epoch):
        test_loader = self.load_test_images()
        self.test(test_loader,epoch)
        

        
if __name__ == "__main__":
    
    artifacts_path = os.path.join(os.path.dirname(__file__), "artifacts")
    path_to_training = os.path.join(os.path.dirname(__file__), "artifacts","training_model.onnx"),
    path_to_eval = os.path.join(os.path.dirname(__file__), "artifacts","eval_model.onnx"),
    path_to_optimizer = os.path.join(os.path.dirname(__file__), "artifacts","optimizer_model.onnx"),
    path_to_checkpoint = os.path.join(os.path.dirname(__file__), "artifacts","checkpoint"),
    model_path = os.path.join(os.path.dirname(__file__), "onnx/inference.onnx")
    path_to_config = os.path.join(os.path.dirname(__file__), "static/conf.json")
    
    obj = Artifacts(
        model_path==model_path,
        artifacts_path=artifacts_path
    )
    obj()
    
    train = Train(
        path_to_training = path_to_training,
        path_to_eval = path_to_eval,
        path_to_optimizer = path_to_optimizer,
        path_to_checkpoint = path_to_checkpoint,
        path_to_model=model_path,
        path_to_config=path_to_config
    )
    
    test = Train(
        path_to_training = path_to_training,
        path_to_eval = path_to_eval,
        path_to_optimizer = path_to_optimizer,
        path_to_checkpoint = path_to_checkpoint,
        path_to_model=model_path,
        path_to_config=path_to_config
    )
    
    for i in range(5):
        train(i)
        test(i)
    
    
