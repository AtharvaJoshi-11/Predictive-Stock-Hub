import torch
import os
from torch import nn 

class Stock_LSTM(nn.Module):
  def __init__(self,input_size,hidden_size,num_layers,output_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
    self.fc = nn.Linear(hidden_size , output_size)

  def forward(self,x:torch.Tensor)->torch.Tensor:
    h0 = torch.zeros(self.num_layers, x.size(0),self.hidden_size).requires_grad_()
    c0 = torch.zeros(self.num_layers, x.size(0),self.hidden_size).requires_grad_()
    out , (hn,cn) = self.lstm(x,(h0.detach(),c0.detach()))
    print(f"Shape of lstm_out:{out.shape}")
    out = self.fc(out[:,-1,:])
    print(f"Shape of fc_out: {out.shape}")
    return out

input_size = 1
hidden_size = 32
output_size = 1
num_layers=2


def load_lstm_model():
  model = Stock_LSTM(input_size,hidden_size,num_layers,output_size)
  PATH = "App\stock_model.pth"
  model.load_state_dict(torch.load(PATH))
  model.eval()
  return model

  


