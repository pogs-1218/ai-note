import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# from coursera lecture.
def load_data():
  rng = np.random.default_rng(2)
  X = rng.random(400).reshape(-1,2)
  X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
  X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
  Y = np.zeros(len(X))
  i=0
  for t,d in X:
    y = -3/(260-175)*t + 21
    if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
      Y[i] = 1
    else:
      Y[i] = 0
    i += 1
  return (X, Y.reshape(-1,1))

def train(device, dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  print(f'data size: {size}')
  for batch, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)

    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), (batch+1) * len(x)
      print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

# Define a model
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer = nn.Sequential(
      nn.BatchNorm1d(2),
      nn.Linear(2, 3),
      nn.Sigmoid(),
      nn.Linear(3, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    logits = self.layer(x)
    return logits

x_train, y_train = load_data()
xt = np.tile(x_train, (1000, 1))
yt = np.tile(y_train, (1000, 1))
train_data = TensorDataset(torch.from_numpy(xt).to(torch.float32), 
                           torch.from_numpy(yt).to(torch.float32))
train_dataloader = DataLoader(train_data, batch_size=64)

device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)
print(f'Using {device} device')

# Create the model
model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10
for t in range(epochs):
  print(f'Epochs {t+1}\n----------------------------')
  train(device, train_dataloader, model, loss_fn, optimizer)
print('Done')
torch.save(model.state_dict(), "model.pth")

test_model = NeuralNetwork()
test_model.load_state_dict(torch.load('model.pth'))
test_model.eval()
x_test = np.array([[200, 13.9], [200, 17]],dtype=np.float32)
x_test = torch.from_numpy(x_test[0].reshape(1, -1))
with torch.no_grad():
  print(x_test)
  pred = test_model(x_test)
  print(f'result: {pred}')
