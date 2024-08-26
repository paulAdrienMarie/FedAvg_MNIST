# rename module

import platform
import onnx
import torch
from Artifacts import Artifacts
import json
from onnxruntime.training.api import CheckpointState, Module, Optimizer
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

assert list(platform.python_version_tuple())[:-1] == ["3", "9"]


class Train:
    def __init__(
        self,
        path_to_training,
        path_to_eval,
        path_to_optimizer,
        path_to_checkpoint,
        path_to_model,
    ):
        self.path_to_training = path_to_training
        self.path_to_eval = path_to_eval
        self.path_to_optimizer = path_to_optimizer
        self.path_to_checkpoint = path_to_checkpoint
        self.path_to_model = path_to_model
        self.model = None

    def load_train_images(self):
        # should be a constant at the top of the module
        batch_size = 64
        train_kwargs = {"batch_size": batch_size}

        # constants
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # root dir as an argument of the class
        dataset = datasets.MNIST("./", train=True, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
        return train_loader

    def load_model(self):
        print("Loading the model")
        # root dir + os.path.join
        self.model = onnx.load_model("./onnx/inference.onnx")

    # pep8 (load_json)
    def loadJson(self, path):
        with open(path) as f:
            return json.loads(f.read())

    # name too long pep8
    def preprocess_images_to_numpy(self, images):
        # root dir + os.path.join + don't use UPPER_CASE inside a class
        PREPROCESS_CONFIG = "./static/conf.json"
        pre = self.loadJson(PREPROCESS_CONFIG)

        transform = transforms.Compose(
            [
                # constants
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[pre["mean"]], std=[pre["std"]]),
            ]
        )

        preprocessed_images = [transform(img) for img in images]
        images_tensor = torch.stack(preprocessed_images)
        return images_tensor.view(images_tensor.size(0), -1).numpy()

    def load_training_modules(self):
        print("Loading training modules")
        state = CheckpointState.load_checkpoint(self.path_to_checkpoint)
        module = Module(self.path_to_training, state, self.path_to_eval, device="cpu")
        optimizer = Optimizer(self.path_to_optimizer, module)
        return module, optimizer, state

    def generate_target_logits(self, y_true):
        return np.int64(y_true)

    def train(self, train_loader, epoch):
        module, optimizer, state = self.load_training_modules()

        module.train()
        losses = []
        # don't use enumerate if you drop the index with _
        for _, (data, target) in enumerate(train_loader):
            forward_inputs = [
                # constant
                data.reshape(len(data), 784).numpy(),
                target.numpy().astype(np.int64),
            ]
            train_loss, _ = module(*forward_inputs)
            print(f"LOSS : {train_loss}")
            optimizer.step()
            module.lazy_reset_grad()
            losses.append(train_loss)

        print(f"Epoch: {epoch+1} - Train Loss: {sum(losses)/len(losses):.4f}")

        CheckpointState.save_checkpoint(state, self.path_to_checkpoint)
        module.export_model_for_inferencing("./onnx/mnist_trained.onnx", ["output"])

    def __call__(self, epoch):
        train_loader = self.load_train_images()
        self.train(train_loader, epoch)


# find a more explicit name
class Test:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model

    def load_test_images(self):
        test_batch_size = 1000
        test_kwargs = {"batch_size": test_batch_size}

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # root dir
        dataset2 = datasets.MNIST("./", train=False, transform=transform)

        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        return test_loader

    def softmax_activation(self, logits):
        return np.argmax(logits, axis=-1)  # Assuming logits is a list of arrays

    def compute_accuracy(self, y_pred, y_true):
        return accuracy_score(y_true, y_pred)

    def test(self, test_loader, epoch):
        # artifact_directory as a class arg?
        state = CheckpointState.load_checkpoint("./artifacts/checkpoint")

        module = Module(
            "./artifacts/training_model.onnx",
            state,
            "./artifacts/eval_model.onnx",
            device="cpu",
        )

        module.eval()
        losses = []
        import evaluate

        metric = evaluate.load("accuracy")

        for _, (data, target) in enumerate(test_loader):
            forward_inputs = [
                data.reshape(len(data), 784).numpy(),
                target.numpy().astype(np.int64),
            ]
            test_loss, logits = module(*forward_inputs)
            metric.add_batch(
                references=target, predictions=self.softmax_activation(logits)
            )
            losses.append(test_loss)

        metrics = metric.compute()
        print(
            f'Epoch: {epoch+1} - Test Loss: {sum(losses)/len(losses):.4f}, Accuracy : {metrics["accuracy"]:.2f}'
        )

    def __call__(self, epoch):
        test_loader = self.load_test_images()
        self.test(test_loader, epoch)


if __name__ == "__main__":
    print("Exporting the model")
    path_to_model = "./onnx/mnist_trained.onnx"
    art = Artifacts(path_to_model=path_to_model)
    art()
    # Training loop
    train = Train(
        "./artifacts/training_model.onnx",
        "./artifacts/eval_model.onnx",
        "./artifacts/optimizer_model.onnx",
        "./artifacts/Checkpoint",
        path_to_model,
    )
    test = Test(path_to_model=path_to_model)

    for i in range(5):
        train(i)
        test(i)
