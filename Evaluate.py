from torchvision import transforms, datasets
import torch
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score
from onnxruntime.training.api import CheckpointState, Module
import random
from torch.utils.data import Subset



class Test:
    def __init__(self):
        self.path_to_model = "./model/inference.onnx"
        self.path_to_trained = "./model/mnist_trained.onnx"
   
    def load_test_images(self):
        batch_size = 64
        train_kwargs = {'batch_size': batch_size}
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST('./', train=False, download=True, transform=transform)

        # Select a random subset of 600 indices
        indices = list(range(len(dataset)))
        random.seed(42)  # For reproducibility
        subset_indices = random.sample(indices, 200)
        subset = Subset(dataset, subset_indices)
        
        test_loader = torch.utils.data.DataLoader(subset, **train_kwargs)
        return test_loader
    
    def load_inference_session(self,path):
        return ort.InferenceSession(path)
    
    def get_predicted_labels(self, logits):
        return np.argmax(logits, axis=-1)  # Assuming logits is a 2D array (batch_size, num_classes)

    def compute_accuracy(self,y_pred,y_true):
        return accuracy_score(y_pred,y_true)
    
    def test_artifacts(self,test_loader):
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
            metric.add_batch(references=target, predictions=self.get_predicted_labels(logits))
            losses.append(test_loss)

        metrics = metric.compute()
        print(f'Test Loss: {sum(losses)/len(losses):.4f}, Accuracy : {metrics["accuracy"]:.4f}')
        
    def test_model(self, test_loader):
        inference_session = self.load_inference_session(self.path_to_model)
        mnist_trained_session = self.load_inference_session(self.path_to_trained)
        input_name = inference_session.get_inputs()[0].name
        output_name = inference_session.get_outputs()[0].name
        
        all_labels = []
        all_targets = []
        all_labels_trained = []
    
        for data, target in test_loader:
            # Run the inference session on the updated model
            forward_inputs = data.reshape(len(data), 784).numpy()
            logits = inference_session.run([output_name], {input_name: forward_inputs})
            labels = self.get_predicted_labels(logits[0])
            all_labels.extend(labels)
            all_targets.extend(target.numpy())
            # Run the inference session on the previously trained model
            logits = mnist_trained_session.run([output_name], {input_name: forward_inputs})
            labels = self.get_predicted_labels(logits[0])
            all_labels_trained.extend(labels)

        accuracy = self.compute_accuracy(all_labels, all_targets)
        print(f"Accuracy with updated model : {accuracy:.4f}")
        print(f"Accuracy with trained model : {self.compute_accuracy(all_labels_trained,all_targets)}")
        
    def __call__(self):
        test_loader = self.load_test_images()
        print("Testing with the training artifacts")
        self.test_artifacts(test_loader)
        print("Testing with the updated model")
        self.test_model(test_loader)
        
if __name__=="__main__":
    test = Test()
    test()
    