#obtain the data
import data

# handeling data
import numpy as np
import pandas as pd
import bentoml

# splitting data
from sklearn.model_selection import train_test_split

# training the data
import torch
from torch import nn
import torchmetrics
from torchmetrics import Accuracy

# scale data
from sklearn.preprocessing import MinMaxScaler


def drop_columns(data_frame, col_name):
  try:
    data_frame = data_frame.drop(columns=col_name)
    print(f'The remaining colums are: {data_frame.columns}')
  except:
    print(f'The column(s) have already been dropped, the remaining are: {data_frame.columns}')
  return data_frame


def split_format_data(data_frame, target):
  # create X and y varialbles
  y = data_frame[target]
  X = data_frame.drop(columns=target)

  # convert y into type int64
  y = torch.tensor(y.values)
  y = y.type(torch.LongTensor)

  #convert X into type float32
  X = torch.tensor(X.values)
  X = X.type(torch.FloatTensor)

  # split the data
  X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=True)
  return X_train, X_valid, y_train, y_valid

def compile_model():
    # create a loss function
    loss_fn = nn.CrossEntropyLoss()

    # create the accuracy function
    accuracy = Accuracy(task='binary', num_classes=2)

    # set learning rate
    LR = 0.1

    # create an optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

    return loss_fn, accuracy, optimizer

def train_model(model, X_train, y_train, X_valid, y_valid, loss_fn, accuracy, optimizer):
  # set device agnostic code
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # set number of epochs
  EPOCHS = 100

  # put the data and the model on the desired device
  model = model.to(device)
  accuracy = accuracy.to(device)
  X_train = X_train.to(device)
  y_train = y_train.to(device)
  X_valid = X_valid.to(device)
  y_valid = y_valid.to(device)

  #loop through the data
  for epoch in range(EPOCHS):

    # set model to train
    model.train()

    # forward pass
    y_logits = model(X_train)

    # convert logits into probabilites then prediciton labels
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # calculate the loss
    loss = loss_fn(y_logits, y_train)

    # calculate the accuracy
    acc = accuracy(y_pred, y_train)

    # optimizer zero_grad
    optimizer.zero_grad()

    # back propagation
    loss.backward()

    # optimizer step
    optimizer.step()


    ### Validation

    # put model into eval mode
    model.eval()

    # create the valid logits
    valid_logits = model(X_valid)

    # convert logits into prediction proabailites then prediction labels
    valid_preds = torch.softmax(valid_logits, dim=1).argmax(dim=1)

    # calculate the loss
    valid_loss = loss_fn(valid_logits, y_valid)

    # calculate the accuracy
    valid_acc = accuracy(valid_preds, y_valid)

    # print out the results
    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Valid Loss: {valid_loss:.5f}, Valid Acc: {valid_acc:.2f}%")


class Model(nn.Module):
  def __init__(self, inputs, outputs, hidden=20):
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=inputs, out_features=hidden),
        nn.ReLU(),
        nn.Linear(in_features=hidden, out_features=hidden),
        nn.ReLU(),
        nn.Linear(in_features=hidden, out_features=outputs)
    )

  def forward(self, x):
    return self.linear_layer_stack(x)

  
  
  
  
  
  
  
  
if __name__ == "__main__":

  # read in the data
  raw_train = pd.read_csv("/workspaces/Model-Deployment-With-Bentoml/data/train.csv")

  # drop id column
  raw_train = drop_columns(raw_train, 'id')
  
  # split the data
  X_train, X_valid, y_train, y_valid = split_format_data(raw_train, 'Class')

  # Instantiate the model
  input_num = 8
  output_num = 2
  model = Model(inputs=input_num, outputs=output_num)

  # obtain loss, accuracy and optimizer
  loss_fn, accuracy, optimizer = compile_model()

  # train the model
  train_model(model=model,
          X_train=X_train,
          y_train=y_train,
          X_valid=X_valid,
          y_valid=y_valid,
          loss_fn=loss_fn,
          accuracy=accuracy,
          optimizer=optimizer)
  
  # save the model to the Bento local model store
  saved_pytorch_model = bentoml.pytorch.save_model(
    'pytorch_model',
    model, 
    signatures={   # model signatures for runner inference
      "predict": {
          "batchable": True,
          "batch_dim": 0,
      }
    })
  print(f'Model saved: {saved_pytorch_model}')

  # Model saved: Model(tag="pytorch_model:xzrlgzsovglrsycf")