import torch
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

pytorch_runner = bentoml.pytorch.get("pytorch_model:latest").to_runner()

svc = bentoml.Service(name="pytorch_model", runners=[pytorch_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
   input_series = torch.tensor(input_series)
   result = pytorch_runner.run(input_series)
   return result