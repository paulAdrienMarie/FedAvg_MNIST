import { run } from "./federated.js";

const launch_federated = document.getElementById("launch_federated");

// launch federated learning
launch_federated.addEventListener("click", async function (e) {
  await run();
  location.reload(true);
});
