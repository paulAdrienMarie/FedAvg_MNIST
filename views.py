from aiohttp import web
import os
from ModelUpdater import ModelUpdater
from Artifacts import Artifacts
from Evaluate import Test

# Initialize ModelUpdater with the path to the ONNX model
here = os.path.dirname(__file__)

updated_model_path = os.path.join(here, "onnx/inference.onnx")
model_updater = ModelUpdater(updated_model_path=updated_model_path)
artifacts = Artifacts("./onnx/inference.onnx")
test = Test()

async def update_model(request):
    try:
        data = await request.json()
        updated_weights = data.get("updated_weights")
        values = updated_weights["cpuData"].values()
        list_values = list(values)
        user_id = data.get("user_id")
        epoch = data.get("epoch")
        
        print(f'Treating data of user {user_id} - Epoch {epoch+1}/50')
        model_updater.update_weights(updated_weights=list_values)
        
        if len(model_updater.fc1_weights)%50 == 0:
            print("Updating the model parameters")
            response_data = model_updater.update_model()
            print("Generating the new training artifacts based on the new model")
            artifacts.gen_artifacts()
            print("Start testing phase")
            test()
        else:
            response_data = {"message": "User data added, waiting for more data to update the model"}
        
        return web.json_response(response_data)
    
    except Exception as e:
        error_message = str(e)
        response_data = {"error": error_message}
        return web.json_response(response_data, status=500)

async def index(request):
    print("New connection")
    return web.FileResponse("./index.html")

async def style(request):
    return web.FileResponse("./style.css")