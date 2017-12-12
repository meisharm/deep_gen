import numpy as np


def classify_point(w_1, w_2,  p, b=0):
    w1_p = np.matmul(w_1, np.transpose(p))
    w2_p = np.matmul(w_2, np.transpose(p))
    cls = min(w1_p, w2_p)

    if cls > b:
        return 1
    #     a = [0,1]
    #     np.ndarray(a)
    #     return a
    # a = [1,0]
    # np.ndarray(a)
    return 0


def generate_points(w_1, w_2, num_samples, dim,b=0):
    points = np.zeros([num_samples, dim])
    labels = np.zeros([num_samples, 1])
    q=0
    while q < num_samples:
        point = np.random.uniform(-1, 1, dim)
        cls = classify_point(w_1, w_2, point)
        cls_b = classify_point(w_1, w_2, point,b)
        if cls==cls_b:


           points[q] = point
           labels[q] = int(cls)
           q=q+1
    return points, labels

def generate_points_round(r, num_samples,c_x=0, c_y=0):
    points = np.zeros([num_samples, 2])
    labels = np.zeros([num_samples, 1])
    q = 0
    while q < num_samples:
        point = np.random.uniform(-1, 1, 2)
        cls = np.add((point[0] - c_x) ** 2, (points[1] - c_y) ** 2)
        cls = np.sum(cls)
        if cls <= r:
            points[q] = point
            labels[q] = 1
            q = q + 1
        if cls > r:
            points[q] = point
            labels[q] = 0
            q = q + 1
    return points, labels

def generate_points_round_tr(r, num_samples,c_x=0, c_y=0):
    points = np.zeros([num_samples, 2])
    labels = np.zeros([num_samples, 1])
    q = 0
    while q < num_samples:
         point = np.random.uniform(-0.1, 0.1, 2)
         point[0] = np.random.uniform(-1, 1)

         cls = np.add((point[0]-c_x)**2,(points[1]-c_y)**2)
         cls=np.sum(cls)
         if cls <= r:
                points[q] = point
                labels[q] = 1
                q = q + 1
         if cls > r:
                points[q] = point
                labels[q] = 0
                q = q + 1
    return points, labels
def generate_points_tr(w_1, w_2, num_samples, dim,b=0):
    points = np.zeros([num_samples, dim])
    labels = np.zeros([num_samples, 1])
    q=0
    while q < num_samples:
        point = np.random.uniform(-0.1, 0.1, dim)
        cls = classify_point(w_1, w_2, point)
        cls_b = classify_point(w_1, w_2, point,b)
        if cls==cls_b:


           points[q] = point
           labels[q] = int(cls)
           q=q+1

    return points, labels

if __name__ == "__main__":
    w_1 = np.random.uniform(0, 1, 3)
    w_2 = np.random.uniform(0, 1, 3)

    points, labels = generate_points(w_1, w_2, 10000, 3)

    mean = np.mean(labels)
    print('Mean is %f' % (mean))
    print('Finished generating ')