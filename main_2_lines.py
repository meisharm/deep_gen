import matplotlib.pyplot as plt

import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

from data import generate_points
from data import generate_points_tr
from data import generate_points_round
from data import generate_points_round_tr


from nets import TwoLayer
plt.interactive(False)
DIM = 2
EPOCHS = 5000
BATCH_SIZE = 128
LR = 1e-3
b=0.2

fc_net = TwoLayer(DIM)
criterion = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(fc_net.parameters(), lr=LR)

w_1 = np.random.uniform(0, 1, DIM)
w_1[0]=np.random.uniform(0.9,1)
w_2 = np.random.uniform(0, 1, DIM)
w_2[0]=np.random.uniform(0.,0.1)



intersection_point_x1=[1,-1*w_1[0]/w_1[1]]
intersection_point_xm1=[-1,1*w_1[0]/w_1[1]]
intersection_point_y1=[-1*w_1[1]/w_1[0],1]
intersection_point_ym1=[1*w_1[1]/w_1[0],-1]
list_1=[intersection_point_xm1, intersection_point_y1, intersection_point_ym1 ,intersection_point_x1]
intersection_point_x2=[1,-1*w_2[0]/w_2[1]]
intersection_point_xm2=[-1,1*w_2[0]/w_2[1]]
intersection_point_y2=[-1*w_2[1]/w_2[0],1]
intersection_point_ym2=[1*w_2[1]/w_2[0],-1]
list_2=[intersection_point_xm2, intersection_point_y2, intersection_point_ym2 ,intersection_point_x2]
def in_box(p, ls):
    for pp, l in zip(p, ls):
        if pp < l[0] or pp > l[1]:
            return False
    return True
list_1 = [i for i in list_1 if in_box(i, [[-1, 1], [-1, 1]])]
list_2 = [i for i in list_2 if in_box(i, [[-1, 1], [-1, 1]])]
def data_for_drow_line(w,b):
    intersection_point_x1 = [1, (b- w[0]) / w[1]]
    intersection_point_xm1 = [-1, (b+ w[0]) / w[1]]
    intersection_point_y1 = [(b- w[1]) / w[0], 1]
    intersection_point_ym1 = [(b+ w[1]) / w[0], -1]
    list_1 = [intersection_point_xm1, intersection_point_y1, intersection_point_ym1, intersection_point_x1]
    list_1 = [i for i in list_1 if in_box(i, [[-1, 1], [-1, 1]])]
    list_1x, list_1y = list(zip(*list_1))
    return list_1x, list_1y
def line_from_two(w_1,_w_2,b_1,b_2):
    a=([w_1,w_2])
    b_eq=([b_1,b_2])
    ineter=np.linalg.solve(a,b_eq)
    point_w1=data_for_drow_line(w_1,b_1)
    point_w2=data_for_drow_line(w_2,b_2)
    point_w1=[point for point in point_w1 if np.matmul(point,np.transpose(w_2)>=b_2)]
    point_w2=[point for point in point_w2 if np.matmul(point,np.transpose(w_1)>=b_1)]
    all_points=point_w1+point_w2
    all_points.append(ineter)
    all_x,all_y=list(zip(*all_points))
    return all_x,all_y

fc_net.train()
def accuracy(pred,lab):
    real_pred = pred.data > 0.
    correct = real_pred == lab.data.byte()
    correct_sum = correct.sum()
    num_of_element = lab.numel()
    acc = correct_sum / float(num_of_element)
    return acc


for i in range(EPOCHS):
    optimizer.zero_grad()

    points, labels = generate_points(w_1,w_2, BATCH_SIZE,2,b=0.2) #small strip
    points = torch.from_numpy(points)
    labels = torch.from_numpy(labels)

    #points, labels = points.type(torch.FloatTensor), labels.type(torch.IntTensor)
    points, labels = points.type(torch.FloatTensor), labels.type(torch.FloatTensor)

    points, labels = Variable(points), Variable(labels)
    predictions = fc_net(points)
    labels = labels.squeeze()
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    acc= accuracy(predictions,labels)

    print("Iteration {} Loss {} Acc {}".format(i+1, loss.data[0], acc))
    if i%100==0:

        view_points, _ = generate_points(w_1,w_2, 100000,2,b=0.2) #all

        p_x_r=[]
        p_y_r=[]
        p_x_g=[]
        p_y_g=[]
        p_x_y=[]
        p_y_y=[]
        p_x_y1=[]
        p_y_y1=[]
        view_points = torch.from_numpy(view_points)

        view_points = view_points.type(torch.FloatTensor)
        view_points = Variable(view_points)
        fc_net.eval()
        a=fc_net(view_points)
        point_pred = a.data > 0.
        for i in range(len(view_points)):
          if point_pred[i]==1:
            p_x_y.append(view_points[i].data[0])
            p_y_y.append(view_points[i].data[1])
          else:
            p_x_y1.append(view_points[i].data[0])
            p_y_y1.append(view_points[i].data[1])
        list_1x, list_1y = data_for_drow_line(w_1,b/2)
        list_2x, list_2y = data_for_drow_line(w_2,b/2)
# list_3x, list_3y = data_for_drow_line(w_1,0.1)
# list_4x, list_4y = data_for_drow_line(w_2,0.1)
#all_x,all_y=line_from_two(w_1,w_2,0.1,0.1)

    # plt.plot(p_x_y, p_y_y, 'yo',p_x_y1, p_y_y1, 'yo')
    # plt.axis([-1, 1, -1, 1])
    # plt.show()
        view_points, view_lab = generate_points(w_1,w_2, 100000,2,b=0) #all

        p_x_r=[]
        p_y_r=[]
        p_x_g=[]
        p_y_g=[]
        view_points = torch.from_numpy(view_points)

        view_points = view_points.type(torch.FloatTensor)
        view_points = Variable(view_points)
        fc_net.eval()
        a=fc_net(view_points)
        point_pred = a.data > 0.
        for i in range(len(view_points)):
          if point_pred[i]==1:
            p_x_r.append(view_points[i].data[0])
            p_y_r.append(view_points[i].data[1])
          else:
            p_x_g.append(view_points[i].data[0])
            p_y_g.append(view_points[i].data[1])
        list_1x, list_1y = data_for_drow_line(w_1,b/2)
        list_2x, list_2y = data_for_drow_line(w_2,b/2)
# list_3x, list_3y = data_for_drow_line(w_1,0.1)
# list_4x, list_4y = data_for_drow_line(w_2,0.1)
#all_x,all_y=line_from_two(w_1,w_2,0.1,0.1)

        plt.plot(p_x_r, p_y_r)#, 'ro',p_x_g, p_y_g, 'go',p_x_y, p_y_y, 'yo',p_x_y1, p_y_y1, 'yo'
        plt.axis([-1, 1, -1, 1])
        plt.show()
        view_points, view_lab = generate_points(w_1,w_2, 100000,2,b=0) #all

        p_x_r=[]
        p_y_r=[]
        p_x_g=[]
        p_y_g=[]
        view_points = torch.from_numpy(view_points)

        view_points = view_points.type(torch.FloatTensor)
        view_points = Variable(view_points)
        fc_net.eval()
        a=fc_net(view_points)
        point_pred = a.data > 0.
        for i in range(len(view_points)):
          if point_pred[i]==1:
            p_x_r.append(view_points[i].data[0])
            p_y_r.append(view_points[i].data[1])
          else:
            p_x_g.append(view_points[i].data[0])
            p_y_g.append(view_points[i].data[1])
        list_1x, list_1y = data_for_drow_line(w_1,b/2)
        list_2x, list_2y = data_for_drow_line(w_2,b/2)
# list_3x, list_3y = data_for_drow_line(w_1,0.1)
# list_4x, list_4y = data_for_drow_line(w_2,0.1)
#all_x,all_y=line_from_two(w_1,w_2,0.1,0.1)

        plt.plot(p_x_r, p_y_r, 'ro',p_x_g, p_y_g, 'go',list_1x, list_1y,'b-',list_2x, list_2y,'b-')
        plt.axis([-1, 1, -1, 1])
        plt.show()