import base64
import json
from io import BytesIO
import os
from Artifacts import Artifacts
from datasets import load_dataset

# Constants
NUM_THREADS = 20

class FederatedPreparer:
    """
    Prepare the Federated Learning scenario by doing the following :
    
        - Generate nb_users json files that contain a subset of the MNIST dataset as follows :
            - List[Dict{label,base64}]
        - Generate the training artifacts using the Artifacts class.
    
    Attributes:
    nb_users -- Number of users for the Federated Learning
    batch_size -- Size of the subset to attribute to one client 
    """
    
    def __init__(self, nb_users, batch_size):
        """
        Initializes a new FederatedPreparer instance.
        
        Arguments:
        nb_users -- Number of users for the Federated Learning
        batch_size -- Size of the subset to attribute to one client
        """
        
        self.nb_users = nb_users
        self.batch_size = batch_size

    def generate(self):
        """
        Convert images of the training set of the MNIST dataset in 
        string using base64 encoding.
        """
        
        DATASET = "ylecun/mnist"
        ds = load_dataset(DATASET)
        images = ds["train"]["image"]
        labels = ds["train"]["label"]
        dataset = []
        
        for img in images:
            dataset.append(self.image_message(img)["url"])
        
        return dataset, labels

    def image_message(self, image):
        """
        Generates the string representation of the given image using base64 encoding.
        
        Arguments:
        image -- The image to process.
        """
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte).decode("utf-8")
        encoded_image = f"data:image/png;base64,{img_base64}"
        return {"url": encoded_image}
    
    def prepare_jsons_for_federated_learning(self, images, labels):
        """
        Prepare the JSON files for federated learning directly from images and labels.
        
        Arguments:
        images -- List of base64 encoded images
        labels -- List of corresponding labels
        """
        
        output_dir = "./static/dataset/"
        os.makedirs(output_dir, exist_ok=True)

        # Clear the output directory if it is not empty
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove the directory (only if empty)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        for user_id in range(self.nb_users):
            start_index = user_id * self.batch_size
            end_index = start_index + self.batch_size
            
            if start_index >= len(labels):
                break
            
            print(f"Creating JSON file for user {user_id + 1}")
            
            pictures_labels = labels[start_index:end_index]
            pictures_base64 = images[start_index:end_index]
            
            ds = []
            for i, label in enumerate(pictures_labels):
                ds.append({
                    "label": label,
                    "base64": pictures_base64[i]
                })
            
            output_file = os.path.join(output_dir, f"user_{user_id + 1}.json")
            print(f"Saving the set of images in {output_file}")
            
            with open(output_file, "w") as f:
                json.dump(ds, f)

    def prepare_training_artifacts(self):
        """
        Prepare the training artifacts.
        """
        
        obj = Artifacts(
            model_path="./onnx/inference.onnx",
            artifacts_path="./artifacts"
        )
        obj()
        
    
    def __call__(self):
        """
        Launch json files and training artifacts generation.  
        """
        
        images, labels = self.generate()
        print(f"Total images processed: {len(images)}")
            
        self.prepare_jsons_for_federated_learning(images, labels)
        self.prepare_training_artifacts()
        
if __name__ == "__main__":
    preparer = FederatedPreparer(nb_users=100, batch_size=600)
    preparer()
