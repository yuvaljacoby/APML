import numbers

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy import signal

# define the functions we would like to predict:
num_of_functions = 3
size = 4
W = 4 * (np.random.random((size, size)) - 0.5)
y = {
    0: lambda x: np.sum(np.dot(x, W), axis=1),
    1: lambda x: np.max(x, axis=1),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x, W)), axis=1))
}


def l2_loss(y_hat, y, lamb, wights):
    if isinstance(wights, numbers.Number):
        wights = [wights]

    reg_term = np.sum([(lamb / 2) * np.square(numpy.linalg.norm(w)) for w in wights])
    return np.mean(np.square(y_hat - y) + reg_term)



def learn_linear(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a linear model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (w, training_loss, test_loss):
            w: the weights of the linear model
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    w = {func_id: np.zeros(size) for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):
        x_test, y_test = X['test'], Y[func_id]['test']
        for _ in range(iterations):
            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx, :], Y[func_id]['train'][idx]

            # calculate the loss and derivatives:
            p = np.dot(x, w[func_id])
            loss = l2_loss(p, y, lamb, w[func_id])
            iteration_test_loss = l2_loss(np.dot(x_test, w[func_id]), y_test, lamb, w[func_id])
            dl_dp = 2 * (p - y)
            dl_dw = np.mean(np.matlib.repmat(dl_dp.reshape((batch_size, 1)), 1, size) * x, axis=0)  # + lamb*w[func_id]

            # update the model and record the loss:
            w[func_id] -= learning_rate * dl_dw
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return w, training_loss, test_loss


def forward(cnn_model, x):
    """
    Given the CNN model, fill up a dictionary with the forward pass values.
    :param cnn_model: the model
    :param x: the input of the CNN
    :return: a dictionary with the forward pass values
    """

    fwd = {}
    fwd['x'] = x  # (32,4)
    fwd['o1'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [np.array(cnn_model['w1'])], mode='same'))
    fwd['o2'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [cnn_model['w2']], mode='same'))  # (32,4)
    fwd['m'] = np.array([np.max(fwd['o1'][:, :2], axis=1),
                         np.max(fwd['o1'][:, 2:], axis=1),
                         np.max(fwd['o2'][:, :2], axis=1),
                         np.max(fwd['o2'][:, 2:], axis=1)])  # (4, 1)
    fwd['m_argmax'] = np.array([np.argmax(fwd['o1'][:, :2], axis=1),
                                np.argmax(fwd['o1'][:, 2:], axis=1) + 2,
                                np.argmax(fwd['o2'][:, :2], axis=1),
                                np.argmax(fwd['o2'][:, 2:], axis=1) + 2])  # (4, 1)
    fwd['p'] = np.dot(cnn_model['u'], fwd['m'])  # (1,)

    return fwd


def backprop(model, y, fwd, batch_size):
    """
    given the forward pass values and the labels, calculate the derivatives
    using the back propagation algorithm.
    :param model: the model
    :param y: the labels
    :param fwd: the forward pass values
    :param batch_size: the batch size
    :return: a tuple of (dl_dw1, dl_dw2, dl_du)
            dl_dw1: the derivative of the w1 vector
            dl_dw2: the derivative of the w2 vector
            dl_du: the derivative of the u vector
    """
    dl_dp = 2 * (fwd['p'] - y)  # (32,)
    dp_dm = model['u']  # (4,)
    dl_dm = np.multiply(dl_dp[:, None], dp_dm)  # (32,4)

    dl_do1 = np.zeros((batch_size, size))
    dl_do2 = np.zeros((batch_size, size))
    dl_do1[np.arange(batch_size), fwd['m_argmax'][0, :]] = dl_dm[:, 0]
    dl_do1[np.arange(batch_size), fwd['m_argmax'][1, :]] = dl_dm[:, 1]
    dl_do2[np.arange(batch_size), fwd['m_argmax'][2, :]] = dl_dm[:, 2]
    dl_do2[np.arange(batch_size), fwd['m_argmax'][3, :]] = dl_dm[:, 3]

    x = np.insert(np.insert(fwd['x'], 0, 0, axis=0), len(fwd['x']) + 1, 0, axis=0)
    x_mat = np.array([np.array([x[i - 1], x[i], x[i + 1]]).T for i in range(1, len(x) - 1)])
    do1_dw = x_mat
    do2_dw = x_mat
    do1_dw[fwd['o1'] <= 0, :] = np.zeros(3)
    do2_dw[fwd['o2'] <= 0, :] = np.zeros(3)

    dl_dw1 = np.mean(np.multiply(dl_do1[:, :, None], do1_dw).sum(axis=1), axis=0)
    dl_dw2 = np.mean(np.multiply(dl_do2[:, :, None], do2_dw).sum(axis=1), axis=0)
    dl_du = np.mean(np.multiply(np.matlib.repmat(dl_dp, size, 1), fwd['m']), axis=1)

    return (dl_dw1, dl_dw2, dl_du)


def learn_cnn(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a cnn model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (models, training_loss, test_loss):
            models: a model for every function (a dictionary for the parameters)
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    models = {func_id: {} for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):

        # initialize the model:
        models[func_id]['w1'] = np.random.normal(0, 1, size - 1)
        models[func_id]['w2'] = np.random.normal(0, 1, size - 1)
        models[func_id]['u'] = np.random.normal(0, 1, size)

        # train the network:
        for _ in range(iterations):
            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx, :], Y[func_id]['train'][idx]

            # calculate the loss and derivatives using back propagation:
            fwd = forward(models[func_id], x)
            loss = l2_loss(fwd['p'], y, lamb, [models[func_id]['w1'], models[func_id]['w2'], models[func_id]['u']])
            dl_dw1, dl_dw2, dl_du = backprop(models[func_id], y, fwd, batch_size)

            # record the test loss before updating the model:
            test_fwd = forward(models[func_id], X['test'])
            iteration_test_loss = l2_loss(test_fwd['p'], Y[func_id]['test'], lamb,
                                          [models[func_id]['w1'], models[func_id]['w2'], models[func_id]['u']])

            # update the model using the derivatives and record the loss:
            models[func_id]['w1'] -= learning_rate * dl_dw1
            models[func_id]['w2'] -= learning_rate * dl_dw2
            models[func_id]['u'] -= learning_rate * dl_du
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return models, training_loss, test_loss


def plot_loss(training_log, test_loss, title, only_test=True):
    i = 0
    plt.figure()
    for k in test_loss:
        fig_title = title + ' loss, func id: ' + str(k)
        legend = []
        if not only_test:
            plt.plot(training_log[k])
            legend.append('train')
        plt.plot(test_loss[k])
        legend.append('test')
        plt.title(fig_title)
        plt.legend(legend)
        i += 1
        plt.ylabel('loss')
        plt.xlabel('epoch')

        # plt.title(title + " loss")
        plt.tight_layout()
        plt.show()
        plt.savefig("plots/" + fig_title.replace(" ", "_"), )


if __name__ == '__main__':
    # generate the training and test data, adding some noise:
    X = dict(train=5 * (np.random.random((1000, size)) - .5),
             test=5 * (np.random.random((200, size)) - .5))
    Y = {i: {
        'train': y[i](X['train']) * (
                1 + np.random.randn(X['train'].shape[0]) * .01),
        'test': y[i](X['test']) * (
                1 + np.random.randn(X['test'].shape[0]) * .01)}
        for i in range(len(y))}

    w, training_loss, test_loss = learn_linear(X, Y, 1024, 0.5, 750, 10**-3)
    plot_loss(training_loss, test_loss, 'linear model', False)

    w, training_loss, test_loss = learn_cnn(X, Y, 1024, 0.5, 750, 10**-5)
    plot_loss(training_loss, test_loss, 'cnn model', False)
