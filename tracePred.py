import torch as torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

data = pd.read_table("nc180vali/Out/trace.out",header = 0, delim_whitespace = True)
df = data[['UXB']]
df['UXB-1'] = df['UXB'].shift(1)
df['UXB-2'] = df['UXB'].shift(2)
df['UXB-3'] = df['UXB'].shift(3)
print(df)
print(df.drop(index=df.index[[0,1,2]]))
df = df.drop(index=df.index[[0,1,2]])
df_train = df[:1500]
df_test = df[1500:]

train_X = torch.from_numpy(df_train.drop(columns=['UXB'], axis=1).values).float()
train_Y = torch.from_numpy(df_train['UXB'].values).float()
test_X = torch.from_numpy(df_test.drop(columns=['UXB'], axis=1).values).float()
test_Y = torch.from_numpy(df_test['UXB'].values).float()
print(train_X)
print(test_X)

#インプットとアウトプットの数
INPUT_FEATURES = 3
OUTPUT_FEATURES = 1
HIDDEN_NODE = 50
#活性化関数
activation = nn.Tanh()

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.net1 = nn.Linear(INPUT_FEATURES, HIDDEN_NODE)
		self.net2 = nn.Linear(HIDDEN_NODE, HIDDEN_NODE)
		self.net3 = nn.Linear(HIDDEN_NODE, OUTPUT_FEATURES)

	def forward(self, x):
		x = F.relu(self.net1(x))
		x = F.relu(self.net2(x))
		return self.net3(x)

######学習###########################################3
model = Network()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

EPOCHS = 2000
for epoch in tqdm(range(EPOCHS)):
  optimizer.zero_grad()
  outputs = model(train_X)
  loss = criterion(outputs, train_Y)
  loss.backward()
  optimizer.step()

  if epoch % 100 == 99:
    print(f'epoch: {epoch+1:4}, loss: {loss.data}')
print('finished')

torch.save(model.state_dict(), 'test.model')

####検証########################################################
# モデル定義
model = Network()
# パラメータの読み込み
param = torch.load('test.model')
model.load_state_dict(param)
# 評価モードにする
model = model.eval()
outputs = model(test_X)
loss = criterion(outputs, test_Y)
print(outputs)
print(f'loss: {loss}')
