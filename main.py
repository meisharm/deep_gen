import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

from data import generate_points
from nets import TwoLayer

DIM = 1000
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-2

fc_net = TwoLayer(DIM)
criterion = torch.nn.BCELoss()

optimizer = optim.SGD(fc_net.parameters(), lr=LR)

w_1 = np.random.uniform(0, 1, DIM)
w_2 = np.random.uniform(0, 1, DIM)

for i in range(EPOCHS):
    fc_net.train()
    points, labels = generate_points(w_1, w_2, BATCH_SIZE, DIM)
    points = torch.from_numpy(points)
    labels = torch.from_numpy(labels)

    points, labels = points.type(torch.FloatTensor), labels.type(torch.IntTensor)
    points, labels = Variable(points), Variable(labels)
    predictions = fc_net(points)

    optimizer.zero_grad()
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    print("Iteration %d: Loss %f", (i+1, loss))