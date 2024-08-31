
import torch
import numpy as np
import torch.nn as nn



class DPCNN(nn.Module):
    def __init__(self, inputsize,embedding_dim,hidden_dim,out_size,batch_size=1,n_layer = 1, dropout = 0,
                 embedding = None):
        super(DPCNN, self).__init__()    #   这里的5是指输入的特征数
        # self.vocab_size = vocab_size
        # self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.out_shape = out_size
        # self.embedding = embedding
        # self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.conv = nn.Conv1d(in_channels=5, out_channels=5,kernel_size=1)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=1)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.padding1 = nn.ZeroPad2d((1, 1, 0, 0))  # top bottom
        self.relu = nn.ReLU()
        # self.weight_W = nn.Parameter(torch.Tensor(batch_size, hidden_dim, hidden_dim))
        # self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim, n_layer))
        self.firstLayer = nn.Sequential(nn.Linear(inputsize, 5),  # Loan_application_Configuration4
                                        # nn.ReLU(),
                                        # nn.Linear(45, 30),
                                        # nn.ReLU(),
                                        # nn.Linear(30, 5),
                                        # nn.ReLU()
                                        )
        # self.first2Layer = nn.Sequential(nn.Linear(5, 45),  # Loan_application_Configuration4
        #                                 nn.ReLU(),
        #                                 nn.Linear(45, 30),
        #                                 nn.ReLU(),
        #                                 nn.Linear(30, 5),
        #                                 nn.ReLU()
        #                                 )
        # self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_dim, dropout=self.dropout,
        #                   num_layers=self.n_layer, bidirectional=False)
        # self.rnn = nn.RNN(input_size=1, hidden_size=hidden_dim, dropout=self.dropout,
        #                   num_layers=self.n_layer, bidirectional=False)
        # self.first2Layer = nn.Sequential(nn.Linear(hidden_dim, 45),  # Loan_application_Configuration4
        #                                 nn.ReLU(),
        #                                 nn.Linear(45, 30),
        #                                 nn.ReLU(),
        #                                 nn.Linear(30, 5),
        #                                 nn.ReLU()
        #                                 )
        # self.secondLayer = nn.Sequential(nn.Linear(hidden_dim, 45),  # Loan_application_Configuration4
        #                                 nn.ReLU(),
        #                                 nn.Linear(45, 30),
        #                                 nn.ReLU(),
        #                                 nn.Linear(30, hidden_dim),
        #                                 nn.ReLU()
        #                                 )

        self.out = nn.Linear(3, 1)



    def forward(self, x):
        first = self.firstLayer(x)
        # first = self.firstLayer(first0)
        # first=self.padding1(first)
        input = first.view(9, 5, 1)
        # input=self.padding1(input)
        input=self.relu(input)
        input=self.conv(input)
        input=self.relu(input)
        input=self.conv(input)
        input=input.squeeze()
        input=self.max_pool(input)
        input = input.view(9, 3, 1)
        input = self.relu(input)
        input = self.conv1(input)
        input = self.relu(input)
        input = self.conv1(input)
        input = input.squeeze()

        output = self.out(input)
        # output = self.secondLayer(hn0)
        return output
