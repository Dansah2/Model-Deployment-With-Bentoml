import bentoml

runner = bentoml.pytorch.get("pytorch_model:xzrlgzsovglrsycf").to_runner()
runner.init_local()
print(runner.predict.run([[133.171875, 59.716081, 0.043133, -0.703383, 54.917224, 70.084438, 0.749798, -0.649512]]))