import base64
import json
from io import BytesIO
import os
from Artifacts import Artifacts
from datasets import load_dataset

# Constants
IMAGES_DIR = "./dest/"
NUM_THREADS = 20

class FederatedPreparer:
    """
    Generates the string representation of a set of images using base64 encoding.
    Aims to be used during a Federated Learning scenario in the web.
    """
    
    def __init__(self, nb_users, batch_size):
        """Initializes a new Base64Generator instance."""
        self.nb_users = nb_users
        self.batch_size = batch_size

    def generate(self):
        """Creates a dictionary {id: string} using the COCO dataset."""
        DATASET = "ylecun/mnist"
        ds = load_dataset(DATASET)
        images = ds["train"]["image"]
        labels = ds["train"]["label"]
        dataset = []
        
        for i, img in enumerate(images):
            dataset.append(self.image_message(img)["url"])
        
        print(len(dataset))
        return dataset, labels

    def image_message(self, image):
        """
        Generates the string representation of the given image using base64 encoding.
        
        image -- The image to process.
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte).decode("utf-8")
        encoded_image = f"data:image/png;base64,{img_base64}"
        return {"url": encoded_image}
    
    def prepare_jsons_for_federated_learning(self):

        with open("./static/train_labels.json") as f:
            train_data = json.loads(f.read())
            
        with open("./static/train_base64images.json") as f:
            train_base64_data = json.loads(f.read())
            
        assert len(train_data) == len(train_base64_data), f"Inconsistent sizes: train_data has {len(train_data)} items, train_base64_data has {len(train_base64_data)} items."

        NUMUSERS = self.nb_users
        BATCHSIZE = self.batch_size

        output_dir = "./static/dataset/"
        os.makedirs(output_dir, exist_ok=True)

        for user_id in range(NUMUSERS):
            start_index = user_id * BATCHSIZE
            end_index = start_index + BATCHSIZE
            
            if start_index >= len(train_data):
                break
            
            print(f"Creating JSON file for user {user_id + 1}")
            
            pictures_labels = train_data[start_index:end_index]
            pictures_base64 = train_base64_data[start_index:end_index]
            
            ds = []
            for i, label in enumerate(pictures_labels):
                if pictures_base64[i] is None:
                    print(f"Warning: Missing base64 data for image ID {id}")
                ds.append({
                    "label": label,
                    "base64": pictures_base64[i]
                })
            
            output_file = os.path.join(output_dir, f"user_{user_id + 1}.json")
            print(f"Saving the set of images in {output_file}")
            
            with open(output_file, "w") as f:
                json.dump(ds, f)

    def prepare_training_artifacts(self):
        
        obj = Artifacts("./model/inference.onnx")
        obj()
        
    
    def __call__(self):
        """Runs the generate function and saves the resulting dict in a JSON file."""
        
        images, labels = self.generate()
        print(len(images))
        with open("./static/train_base64images.json", "w") as f:
            json.dump(images, f)
            
        with open("./static/train_labels.json","w") as f:
            json.dump(labels, f)
            
        self.prepare_jsons_for_federated_learning()
        self.prepare_training_artifacts()
        
if __name__ == "__main__":
    preparer = FederatedPreparer()
    preparer()
