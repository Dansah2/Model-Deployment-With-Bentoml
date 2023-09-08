import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

pytorch_runner = bentoml.pytorch.get("pytorch_model:latest").to_runner()

svc = bentoml.Service(name="keras_model", runners=[icr_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
   result = pytorch_runner.predict.run(input_series)
   return result