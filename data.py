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


def generate_points(w_1, w_2, num_samples, random_dim):
    points = np.zeros([num_samples, random_dim])
    labels = np.zeros([num_samples, 1])

    for i in range(num_samples):
        point = np.random.uniform(-1, 1, random_dim)
        cls = classify_point(w_1, w_2, point)

        points[i] = point
        labels[i] = cls

    return points, labels


if __name__ == "__main__":
    w_1 = np.random.uniform(0, 1, 3)
    w_2 = np.random.uniform(0, 1, 3)

    points, labels = generate_points(w_1, w_2, 10000, 3)

    mean = np.mean(labels)
    print('Mean is %f' % (mean))
    print('Finished generating ')