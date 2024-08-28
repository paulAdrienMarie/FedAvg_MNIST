from torchvision import transforms, datasets
import torch
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score
from onnxruntime.training.api import CheckpointState, Module
import json
import evaluate

class Test:
    """
    
    Tests the updated model after FedAvg
    
    Attributs:
    path_to_model -- Path to model to test
    metrics -- Dictionnary to store the metrics over the communication rounds
    
    """
    
    def __init__(self):
        """Initializes a new instance of the Test class"""
        self.path_to_model = "./onnx/inference.onnx"
        self.metrics = {
            "accuracies": [],
            "losses": []
        }
   
    def load_test_images(self):
        """Loads test set of images from the MNIST dataset"""
        batch_size = 64
        train_kwargs = {'batch_size': batch_size}
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST('./', train=False, download=True, transform=transform)

        test_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
        return test_loader
    
    def load_inference_session(self):
        """Loads Inference Session"""
        return ort.InferenceSession(self.path_to_model)
    
    def get_predicted_labels(self, logits):
        """
        
        Proceeds logits, returns predicted label
        
        Arguments:
        logits -- Raw output of the model
        
        """
        
        return np.argmax(logits, axis=-1)  # Assuming logits is a 2D array (batch_size, num_classes)

    def compute_accuracy(self,y_pred,y_true):
        """
        
        Computes accuracy metric between two lists of labels
        
        Arguments:
        y_pred -- Predictions of the updated model as a list
        y_true -- True labels as a list
        
        """
        
        return accuracy_score(y_pred,y_true)
    
    def test_artifacts(self,test_loader):
        """
        
        Runs test using the training artifacts
        
        Arguments:
        test_loader -- Test set of MNIST dataset
        
        """
        
        state = CheckpointState.load_checkpoint("./artifacts/checkpoint")

        module = Module(
            "./artifacts/training_model.onnx",
            state,
            "./artifacts/eval_model.onnx",
            device="cpu"
        )
    
        module.eval() # set the module in evaluation mode
        losses = []
        metric = evaluate.load('accuracy')

        for _, (data, target) in enumerate(test_loader):
            forward_inputs = [data.reshape(len(data),784).numpy(),target.numpy().astype(np.int64)]
            test_loss, logits = module(*forward_inputs)
            metric.add_batch(references=target, predictions=self.get_predicted_labels(logits))
            losses.append(test_loss)

        metrics = metric.compute()
        mean_loss = sum(losses)/len(losses)
        accuracy = metrics["accuracy"]
        print(f'Test Loss: {mean_loss:.4f}, Accuracy : {accuracy:.4f}')
        
        self.save_metrics(mean_loss.item(),accuracy)
        
    def save_metrics(self, loss, accuracy):
        """
        
        Save the metrics in a json file
        
        Arguments:
        loss -- Current loss of the model
        accuracy -- Current accuracy of the model
        
        """
        
        self.metrics["losses"].append(loss)
        self.metrics["accuracies"].append(accuracy)
        
        with open("metrics.json","w") as f:
            json.dump(self.metrics,f)
            
    def test_model(self, test_loader):
        """
        
        Runs test using the inference session and the updated model
        
        Arguments:
        test_loader -- Test set of MNIST dataset
        
        """
        
        inference_session = self.load_inference_session(self.path_to_model)
        input_name = inference_session.get_inputs()[0].name
        output_name = inference_session.get_outputs()[0].name
        
        all_labels = []
        all_targets = []
    
        for data, target in test_loader:
            # Run the inference session on the updated model
            forward_inputs = data.reshape(len(data), 784).numpy()
            logits = inference_session.run([output_name], {input_name: forward_inputs})
            labels = self.get_predicted_labels(logits[0])
            all_labels.extend(labels)
            all_targets.extend(target.numpy())

        accuracy = self.compute_accuracy(all_labels, all_targets)
        print(f"Accuracy with updated model : {accuracy:.4f}")
        
    def __call__(self):
        test_loader = self.load_test_images()
        print("Testing with the training artifacts")
        self.test_artifacts(test_loader)
        print("Testing with the updated model")
        self.test_model(test_loader)
        
if __name__=="__main__":
    test = Test()
    test()
    