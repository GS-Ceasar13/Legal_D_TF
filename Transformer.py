import math
import torch
import torch.nn as nn


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# Text classifier based on a pytorch TransformerEncoder.
class Transformer(nn.Module):
    def __init__(
            self,
            inputsize=7,
            nhead=1,
            dim_feedforward=2048,
            num_layers=6,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1
    ):
        super().__init__()
        self.firstLayer = nn.Sequential(nn.Linear(inputsize, 5),  # Loan_application_Configuration4
                                        # nn.ReLU(),
                                        # nn.Linear(45, 30),
                                        # nn.ReLU(),
                                        # nn.Linear(30, 5),
                                        # nn.ReLU()
                                        )
        # vocab_size = NUM_WORDS + 2
        d_model = 5
        # vocab_size, d_model = embeddings.size()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        # Embedding layer definition
        # self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # self.pos_encoder = PositionalEncoding(
        #     d_model=d_model,
        #     dropout=dropout,
        #     vocab_size=vocab_size
        # )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.classifier = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, x):
        # x = self.emb(x) * math.sqrt(self.d_model)
        # x = self.pos_encoder(x)
        first = self.firstLayer(x)
        x = self.transformer_encoder(first)
        # x = x.mean(dim=1)
        x = self.classifier(x)

        return x
