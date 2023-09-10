import bentoml
import torch

runner = bentoml.pytorch.get("pytorch_model:latest").to_runner()
runner.init_local()
test_data = [[133.171875, 59.716081, 0.043133, -0.703383, 54.917224, 70.084438, 0.749798, -0.649512]]
test_data = torch.tensor(test_data)
print(runner.run(test_data))