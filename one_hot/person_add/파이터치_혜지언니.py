# https://medium.com/@inmoonlight/pytorch%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%95%98%EA%B8%B0-intro-afd9c67404c3
# https://github.com/inmoonlight/PyTorchTutorial/blob/master/02_DNN.ipynb

import db_connect as db
import numpy as np
import pandas as pd
import itertools
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data_utils 

def load_data(query, is_train = True):
    query = query
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong','temperature','rain','wind','humidity','person', 'value']
    df = pd.DataFrame(dataset, columns=column_name)
    db.connect.commit()

    if is_train == True:
        # train, test 나누기
        train_value = df[ '2020-09-01' > df['date'] ]
        x = train_value.iloc[:,1:-1]
        y = train_value.iloc[:,-1]
    else:
        test_value = df[df['date'] >=  '2020-09-01']
        x = test_value.iloc[:,1:-1]
        y = test_value.iloc[:,-1]

    
    # 원 핫으로 컬럼 추가해주는 코드!!!!!    
    x = pd.get_dummies(x, columns=["category", "dong"])
    # 카테고리랑 동만 원핫으로 해준다 

    return x, y

x_pred_pd, y_pred_pd = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date ", is_train = False)
x_train_pd, y_train_pd = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date WHERE (d.TIME != 2 AND d.TIME != 3 AND d.TIME != 4 AND d.TIME != 5  AND d.TIME != 6 AND d.TIME != 7 AND d.TIME != 8) ORDER BY DATE, YEAR, MONTH, DAY, TIME, category, dong ASC")

# numpy 배열을 입력받아, Tensor(dtype)로 바꿀때 사용합니다,
# from_numpy로 만들어진 텐서는 해당 ndarray와 메모리를 공유
trn_X  = torch.from_numpy(x_train_pd.astype(float).values)
trn_y  = torch.from_numpy(y_train_pd.astype(float).values)

val_X  = torch.from_numpy(x_pred_pd.astype(float).values)
val_y  = torch.from_numpy(y_pred_pd.astype(float).values)

# for dictionary batch
class Dataset(data_utils.Dataset):

    #__init__()는 클래스의 생성자 역할
    def __init__(self, X, y):  
        self.X = X
        self.y = y
   
    # 인덱스로 접근할 수 있는 이터레이터
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.y[idx]}
   
    def __len__(self):
        return len(self.X)

batch_size=500

# pytorch dataloader
trn = Dataset(trn_X, trn_y)
trn_loader = data_utils.DataLoader(trn, batch_size=batch_size, shuffle=True)

val = Dataset(val_X, val_y)
val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=False)

tmp = next(iter(trn_loader)) # 1. iterator 생성,  2. next 값을 차례대로 꺼내기
# print(tmp)

print("Training Shape", trn_X.shape, trn_y.shape) # Training Shape torch.Size([2459424, 47]) torch.Size([2459424])
print("Testing Shape", val_X.shape, val_y.shape) # Testing Shape torch.Size([177408, 47]) torch.Size([177408])  

num_batches = len(trn_loader)

# Build Model
use_cuda = torch.cuda.is_available()

#defining MLP model
# 언니가 보낸 두번째 링크에 있는 코드
class MLPRegressor(nn.Module):
    
    def __init__(self):
        super(MLPRegressor, self).__init__()
        h1 = nn.Linear(47, 50) # 모양 맞추기
        h2 = nn.Linear(50, 35)
        h3 = nn.Linear(35, 1)
        self.hidden = nn.Sequential(
            h1,
            nn.Tanh(),
            h2,
            nn.Tanh(),
            h3,
        )
        if use_cuda:
            self.hidden = self.hidden.cuda()
        
    def forward(self, x):
        o = self.hidden(x)
        return o.view(-1)

#Train model
model = MLPRegressor()

criterion = nn.MSELoss() # mse로 로스 추출
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
num_batches = len(trn_loader)

print(num_batches)  #38429

trn_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    trn_loss_summary = 0.0
    for i, trn in enumerate(trn_loader):
        trn_X, trn_y = trn['X'], trn['y'] 
        if use_cuda:
            trn_X, trn_y = trn_X.cuda(), trn_y.cuda()
        optimizer.zero_grad()
        trn_pred = model(trn_X.float()) #HERE: why is .float() needed here?
        # https://discuss.pytorch.org/t/pytorch-why-is-float-needed-here-for-runtimeerror-expected-scalar-type-float-but-found-double/98741
        trn_loss = criterion(trn_pred.float(), trn_y.float())
        trn_loss.backward()
        optimizer.step()
        
        trn_loss_summary += trn_loss
        
        if (i+1) % 15 == 0:
            with torch.no_grad():
                val_loss_summary = 0.0
                for j, val in enumerate(val_loader):
                    val_X, val_y = val['X'], val['y']
                    if use_cuda:
                        val_X, val_y = val_X.cuda(), val_y.cuda()
                    val_pred = model(val_X.float())
                    val_loss = criterion(val_pred.float(), val_y.float())
                    val_loss_summary += val_loss
                
            print("epoch: {}/{} | step: {}/{} | trn_loss: {:.4f} | val_loss: {:.4f}".format(
                epoch + 1, num_epochs, i+1, num_batches, (trn_loss_summary/15)**(1/2), (val_loss_summary/len(val_loader))**(1/2)
            ))
                
            trn_loss_list.append((trn_loss_summary/15)**(1/2))
            val_loss_list.append((val_loss_summary/len(val_loader))**(1/2))
            trn_loss_summary = 0.0
        
print("finish Training")

plt.figure(figsize=(16,9))
x_range = range(len(trn_loss_list))
plt.plot(x_range, trn_loss_list, label="trn")
plt.plot(x_range, val_loss_list, label="val")
plt.legend()
plt.xlabel("training steps")
plt.ylabel("loss")
plt.show()





