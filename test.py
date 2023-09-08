import bentoml

pytorch_runner = bentoml.pytorch.get("pytorch_model:latest").to_runner()
pytorch_runner.init_local()
print(pytorch_runner.predict.run([[]]))