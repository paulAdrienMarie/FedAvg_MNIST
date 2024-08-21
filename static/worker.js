import "/dist/tf.min.js";
import * as ort from "/dist/ort.training.wasm.min.js";

// Set up wasm paths
ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

// Initialization of both inference training session
let trainingSession = null;
let inferenceSession = null;

// Number of epochs
let NUMEPOCHS = 5;

// Paths to the training artifacts
const ARTIFACTS_PATH = {
  checkpointState: "/artifacts/checkpoint",
  trainModel: "/artifacts/training_model.onnx",
  evalModel: "/artifacts/eval_model.onnx",
  optimizerModel: "/artifacts/optimizer_model.onnx",
};

// Path to the base model
let MODEL_PATH = "/onnx/inference.onnx";

// Worker code for message handling
self.addEventListener("message", async (event) => {
  let data = event.data;
  let userId = data.userId;
  let epoch = data.epoch;
  let nb_users = data.nb_users;
  let index = getRandomNumber(1,nb_users);
  var user_file = await loadJson(`/script/dataset/user_${index}.json`);

  console.log(`CURRENTLY RUNNING USER ${userId} - EPOCH ${epoch}/50`);
  console.log(`LOADING TRAINING SESSION FOR USER ${userId}`);

  // Load the Training session of the current user
  await loadTrainingSession(ARTIFACTS_PATH);
  let count = 0;
  // Loop over the items of the dataset
  for (const id in user_file) {
    let true_label = user_file[id].label;
    let base64 = user_file[id].base64;
    count ++;
    // Get the label predicted by the base model
    let label = await predict(base64);
    console.log(
      `True label is ${true_label}, ONNX model predicted ${label}`
    );
    // Compare the label predicted by the base model to the true label
    if (true_label !== label) {
      await train(base64, true_label); // Retrain the model on the misclassified image
    }
  }
  // Retrieve the updated weights from the training session
  let params = await trainingSession.getContiguousParameters(true);
  console.log(`Making requests for user ${userId}`);

  // Send the updated weights to the backend server for storage
  let start = Date.now();
  fetch("/update_model", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      updated_weights: params,
      user_id: userId,
      epoch: epoch
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      self.postMessage({
        userId: userId,
      });
      let request_time = Date.now() - start;
      console.log(`Request time : ${request_time} milliseconds`);
      console.log("Model parameters updated");
      console.log(`Request done for user ${userId}`);
    })
    .catch((error) => {
      console.log("Error:", error);
    });
});

self.onerror = function (error) {
  console.error("Worker error:", error);
};

/**
 * Get a random number between min and max
 * @getRandomNumber
 * @param {Int} min - min value
 * @param {Int} max - max value
 * @returns {Int}
 */
function getRandomNumber(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

async function paramsToUint8Buffer(params) {
  let buffer = new ArrayBuffer(params.length * 4);

  let dataView = new DataView(buffer);

  for (let i = 0; i < params.length; i++) {
    dataView.setFloat32(i * 4, params[i], true);
  }

  let parameters = new Uint8Array(buffer);

  return parameters;
}

/**
 * Instantiate an inference session
 * @async
 * @loadInferenceSession
 * @param {String} model_path - Path to the base model
 * @returns {Promise<void>}
 */
async function loadInferenceSession(model_path) {
  console.log("Loading Inference Session");

  try {
    inferenceSession = await ort.InferenceSession.create(model_path);
    console.log("Inference Session successfully loaded");
  } catch (err) {
    console.log("Error loading the Inference Session:", err);
    throw err;
  }
}

/**
 * Instantiate a training session
 * @async
 * @loadTrainingSession
 * @param {Object} training_paths - Paths to the training artifacts
 * @returns {Promise<void>}
 */
async function loadTrainingSession(training_paths) {
  console.log("Trying to load Training Session");

  try {
    trainingSession = await ort.TrainingSession.create(training_paths);
    console.log("Training session loaded");
  } catch (err) {
    console.error("Error loading the training session:", err);
    throw err;
  }
}

/**
 * Loads JSON from a given URL
 * @async
 * @loadJson
 * @param {String} url - URL of the file
 * @returns {Promise<Object>}
 */
async function loadJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error loading JSON", error);
    return null;
  }
}

/**
 * Converts an image in base64 string format into a tensor of shape [1, 784]
 * @async
 * @toTensor
 * @param {String} base64 - base64 encoded representation of the image
 * @returns {Promise<ort.Tensor>}
 */
async function toTensor(base64) {
  const imgBlob = await fetch(base64).then((res) => res.blob());
  const imgBitmap = await createImageBitmap(imgBlob);

  const canvas = new OffscreenCanvas(imgBitmap.width, imgBitmap.height);
  const ctx = canvas.getContext("2d");

  ctx.drawImage(imgBitmap, 0, 0, imgBitmap.width, imgBitmap.height);

  const imageData = ctx.getImageData(0, 0, imgBitmap.width, imgBitmap.height);

  const inputSize = imgBitmap.width * imgBitmap.height; // 784 for MNIST (28x28)
  const dataFromImage = new Float32Array(inputSize);

  for (let i = 0; i < inputSize; i++) {
    const r = imageData.data[i * 4];
    const g = imageData.data[i * 4 + 1];
    const b = imageData.data[i * 4 + 2];

    dataFromImage[i] = (r + g + b) / 3; // Grayscale conversion
  }

  const shape = [1, inputSize];
  const imageTensor = new ort.Tensor("float32", dataFromImage, shape);
  return imageTensor;
}

/**
 * Normalizes input image represented as a tensor
 * @async
 * @preprocessImage
 * @param {ort.Tensor} tensor - Image as a Tensor
 * @returns {Promise<ort.Tensor>}
 */
async function preprocessImage(tensor) {
  const conf = await loadJson("/script/conf.json");
  const imageMean = conf.mean;
  const imageStd = conf.std;

  let data = await tensor.getData();

  data = data.map((value) => (value / 255.0 - imageMean) / imageStd);

  return new ort.Tensor("float32", data, [1, tensor.dims[1]]);
}

/**
 * Performs softmax activation on logits in array format
 * @softmax
 * @param {Array[Float32]} logits - Raw outputs of the onnx model
 * @returns {Array[Float32]} Probability distribution in an array
 */
function softmax(logits) {
  return logits.map((value, index) => {
    return (
      Math.exp(value) /
      logits
        .map((y) => Math.exp(y))
        .reduce((a, b) => a + b)
    );
  });
}

/**
 * Sorts an array in descending order
 * @argsort
 * @param {Array[]} array - The array to be sorted
 * @returns {Promise<Array>} The sorted array
 */
function argsort(array) {
  const arrayWithIndices = Array.from(array).map((value, index) => ({
    value,
    index,
  }));

  arrayWithIndices.sort((a, b) => b.value - a.value);

  return arrayWithIndices.map((item) => item.index);
}

/**
 * Given the base64 image, returns the predicted class
 * @async
 * @predict
 * @param {String} base64 - Base64 representation of the image
 * @returns {Promise<Object>} Predicted class with probability score
 */
async function predict(base64) {
  // Check if the inference session has been loaded
  if (!inferenceSession) {
    await loadInferenceSession(MODEL_PATH);
  }

  const imageTensor = await toTensor(base64);
  const preprocessedImage = await preprocessImage(imageTensor);

  const feeds = {
    "input": preprocessedImage
  };
  const results = await inferenceSession.run(feeds);
  const logits = results.output.cpuData;

  const prob = softmax(logits);
  const top5Classes = argsort(prob).slice(0, 1);

  const label = top5Classes[0];

  return label;
}

/**
 * Perform one training step with a given image and its correct label
 * @async
 * @train
 * @param {String} base64 - Base64 representation of the image
 * @param {String} true_label - Correct label of the image
 * @returns {Promise<void>}
 */
async function train(base64, true_label) {
  // Check if the training session has been loaded
  if (!trainingSession) {
    await loadTrainingSession(ARTIFACTS_PATH);
  }

  const imageTensor = await toTensor(base64);
  const preprocessedImage = await preprocessImage(imageTensor);

  const startTrainingTime = Date.now();
  console.log("Training started");

  for (let epoch = 0; epoch < NUMEPOCHS; epoch++) {
    await runTrainingEpoch(preprocessedImage, epoch, true_label);
  }

  const trainingTime = Date.now() - startTrainingTime;
  console.log(`Training completed in ${trainingTime} milliseconds`);
}


/**
 * Runs a single epoch of the training loop
 * @runTrainingEpoch
 * @param {Set[Tensor]} images - Set of augmented images of the image to train on
 * @param {Number} epoch - Current epoch
 * @param {Tensor} target_tensor - The target tensor
 */
async function runTrainingEpoch(image, epoch, y_true) {
  const epochStartTime = Date.now();
  const lossNodeName = trainingSession.handler.outputNames[0];

  console.log(
    `TRAINING | Epoch ${epoch + 1} / ${NUMEPOCHS} | Starting Training ... `
  );
  const data = new BigInt64Array([BigInt(y_true)]);
  const shape = [1]; // Shape of the tensor (one element)

  // Create the tensor with the int64 data type
  const tensor = new ort.Tensor('int64', data, shape);
  const feeds = {
    "input": image,
    "labels": tensor,
  };
  const results = await trainingSession.runTrainStep(feeds);
  const loss = results[lossNodeName].data;

  console.log(`LOSS: ${loss}`);

  await trainingSession.runOptimizerStep();
  await trainingSession.lazyResetGrad();

  const res = await trainingSession.runEvalStep(feeds);
  console.log("Run eval step", res);

  const epochTime = Date.now() - epochStartTime;
  console.log(`Epoch ${epoch + 1} completed in ${epochTime} milliseconds.`);
}
