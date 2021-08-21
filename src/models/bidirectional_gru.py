import torch
from torch import nn


class BGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix):
        super(BGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], _weight=embedding_matrix)
        self.gru = nn.GRU(400, 10, num_layers=10, bidirectional=True, dropout=0.2, batch_first=False)
        self.avg_pooling = nn.AvgPool2d(3)
        self.max_pooling = nn.MaxPool2d(3)
        self.drop_out = nn.Dropout(0.2)
        self.linear = nn.Linear(336, 19)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).float()  # 85 x 400
        out, h_state = self.gru(x)  # 85 x 20
        out1 = self.avg_pooling(out)  # 28 x 6
        out2 = self.max_pooling(out)  # 28 x 6
        out = torch.cat((out1, out2), 2)  # 28 x 12
        out = out.view(-1, 336)  # 336
        out = self.drop_out(out)  # 336
        out = self.linear(out)  # 336 X 19
        return self.sigmoid(out)
        # return out

    def fit(self, train_data_loader, criterion, optimizer, embedding_matrix, epochs):
        # return trained model
        model = self
        model.train()
        for epoch in range(4):  # loop over the dataset multiple times
            loss_list = []
            for i, data in enumerate(train_data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                loss_list.append(loss.item())

        return model

    def predict(self, test_data_loader):
        self.eval()
        for i, data in enumerate(test_data_loader, 0):
            inputs, labels = data

            outputs = self(inputs)
            # TODO  return labels and outputs
