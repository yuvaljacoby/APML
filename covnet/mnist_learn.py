import pickle

import numpy as np
from keras import utils as keras_utils
from keras.datasets import mnist
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout, Input, UpSampling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adadelta, Adam
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


#
# def evalute_model(x_test, y_test, model, flatten=False):
#     if flatten:
#         x_test = flatten_x(x_test)
#     score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
#     return score

def plot_models(model_logs, model_names, title, only_val=True):
    val_name = "_validation"
    if only_val == False:
        legend = [item for subset in [[name, name + val_name] for name in model_names] for item in subset]
    else:
        legend = [name + val_name for name in model_names]

    plt.figure()
    plt.subplot(2, 1, 1)
    for i in range(len(model_logs)):
        log = model_logs[i]
        if only_val == False:
            plt.plot(log.history['acc'])
        plt.plot(log.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(legend)
    plt.title(title)

    plt.subplot(2, 1, 2)
    for i in range(len(model_logs)):
        log = model_logs[i]
        if only_val == False:
            plt.plot(log.history['loss'])
        plt.plot(log.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legend)
    # plt.title(title + " loss")
    plt.savefig(("plots2/" + str(title) + str(model_names[i])).replace(' ', '_').replace("0.", ""))
    plt.tight_layout()
    # plt.show()


# def plot_model(model_log, model_name):
#     plt.figure()
#     plt.subplot(2, 1, 1)
#     plt.plot(model_log.history['acc'])
#     plt.plot(model_log.history['val_acc'])
#     plt.title(model_name + ' accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'])
#
#     plt.subplot(2, 1, 2)
#     plt.plot(model_log.history['loss'])
#     plt.plot(model_log.history['val_loss'])
#     plt.title(model_name + 'loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper right')
#     plt.tight_layout()
#     plt.show()


def train_model(x_train, y_train, x_test, y_test, model, opt=Adadelta()):
    model.compile(optimizer=opt,
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

    model_log = model.fit(x_train, y_train, epochs=20, batch_size=1024, verbose=2,
                          validation_data=(x_test, y_test))

    # plot_model(model_log, model_name)
    return model, model_log


def flatten_x(X):
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


# def multi_train_multi_layer_linear_model(x_train, y_train, x_test, y_test, classes):
#     num_classes = len(classes)
#     x_train = flatten_x(x_train)
#     x_test = flatten_x(x_test)
#
#     models = [
#         Sequential(
#             [Dense(256, input_shape=(x_train.shape[1],)), Activation('softmax'),
#              Dense(16), Activation('softmax'),
#              Dense(num_classes), Activation('softmax')]),
#
#         Sequential(
#             [Dense(256, input_shape=(x_train.shape[1],)), Activation('softmax'),
#              Dense(32), Activation('softmax'),
#              Dense(num_classes), Activation('softmax')]),
#
#         # .49 after 20
#         Sequential(
#             [Dense(256, input_shape=(x_train.shape[1],)), Activation('softmax'),
#              Dense(64), Activation('softmax'),
#              Dense(num_classes), Activation('softmax')]),
#
#         # .466 after 20 epochs
#         Sequential(
#             [Dense(256, input_shape=(x_train.shape[1],)), Activation('softmax'),
#              Dense(64), Activation('softmax'),
#              Dense(64), Activation('softmax'),
#              Dense(64), Activation('softmax'),
#              Dense(num_classes), Activation('softmax')]),
#     ]
#
#     logs = []
#     for i in range(len(models)):
#         _, log = train_model(x_train, y_train, x_test, y_test, models[i], opt=SGD(lr=0.001))
#         logs.append(log)
#
#         plot_models([log], ["mlp num: " + str(i)], 'Compare between mlp models ' + str(i), False)


def train_multi_layer_linear_model(x_train, y_train, x_test, y_test, num_classes, opt):
    x_train = flatten_x(x_train)
    x_test = flatten_x(x_test)
    model = Sequential([Dense(256, input_shape=(x_train.shape[1],)), Activation('relu'),
                        Dense(16), Activation('relu'),
                        Dense(num_classes), Activation('softmax')])

    return train_model(x_train, y_train, x_test, y_test, model, opt=opt)


def train_linear_model(x_train, y_train, x_test, y_test, num_classes, opt):
    x_train = flatten_x(x_train)
    x_test = flatten_x(x_test)
    model = Sequential([
        # Dense(256, input_shape=(x_train.shape[1],)),
        # Activation('softmax'),
        Dense(num_classes),
        Activation('softmax'),
    ])

    return train_model(x_train, y_train, x_test, y_test, model, opt=opt)


def train_covnet(x_train, y_train, x_test, y_test, num_classes, opt=SGD(lr=0.1)):
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return train_model(x_train, y_train, x_test, y_test, model, opt=opt)


def compare_models(x_train, y_train, x_test, y_test, classes, from_pickle=True):
    opt = Adadelta()
    if from_pickle:
        models = pickle.load(open("models.pickle", "rb"))
        linear_model, linear_log = (models['linear']['model'], models['linear']['log'])
        mlp_model, mlp_log = (models['mlp']['model'], models['mlp']['log'])
        covnet_model, covnet_log = (models['covnet']['model'], models['covnet']['log'])
    else:
        linear_model, linear_log = train_linear_model(x_train, y_train, x_test, y_test, len(classes), opt=opt)
        mlp_model, mlp_log = train_multi_layer_linear_model(x_train, y_train, x_test, y_test, len(classes), opt=opt)
        covnet_model, covnet_log = train_covnet(x_train, y_train, x_test, y_test, len(classes), opt=opt)

        pickle.dump({'linear': {'log': linear_log, 'model': linear_model},
                     'mlp': {'log': mlp_log, 'model': mlp_log},
                     'covnet': {'log': covnet_log, 'model': covnet_model}
                     }, open("models.pickle", "wb"))

    plot_models([linear_log, mlp_log, covnet_log], ['linear', 'mlp', 'covnet'], 'Compare between models', True)
    # plot_models([linear_log, mlp_log], ['linear', 'mlp'], 'Compare between models', False)


def hyper_parameter_covnet(x_train, y_train, x_test, y_test, classes, from_pickle=True):
    lrs = [10**-10, 10**-5, 10**-3, 10**-2]
    models = []
    logs = []
    legend = []
    if from_pickle:
        pickle_info = pickle.load(open("covnet_hyperparameter.pickle", "rb"))
        for k, v in pickle_info.items():
            legend.append(str(k))
            logs.append(v['log'])
            models.append(v['model'])
    else:
        for lr in lrs:
            model, log = train_covnet(x_train, y_train, x_test, y_test, len(classes), opt=SGD(lr))
            logs.append(log)
            models.append(model)
            legend.append("covnet lr:" + str(lr))

        pickle.dump({str(legend[i]): {'log': logs[i], 'model': models[i]} for i in range(len(legend))},
                    open("covnet_hyperparameter.pickle", "wb"))

    plot_models(logs, legend, 'Covnet hyperparameter - learning rate', True)


def plot_decode_encode(encoder, autoencoder, x_test):
    num_images = 10
    random_idx = np.random.randint(len(x_test), size=num_images)
    random_images = x_test[random_idx, :]
    encode_img = encoder.predict(random_images)
    decode_img = autoencoder.predict(random_images)

    plt.figure(figsize=(18, 4))

    for i in range(len(random_images)):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(random_images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(encode_img[i].reshape(2, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(decode_img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def pca_loss(x_train, x_test):
    p = PCA(n_components=2)
    p.fit(x_train)

    train_reconstruct = p.inverse_transform(p.transform(x_train))
    test_reconstruct = p.inverse_transform(p.transform(x_test))
    train_score = ((x_train - train_reconstruct) ** 2).mean()
    test_score = ((x_test - test_reconstruct) ** 2).mean()

    return train_score, test_score, p


def train_autoencoder(x_train, x_test):
    # x_train = flatten_x(x_train)
    # x_test = flatten_x(x_test)
    autoencoder = Sequential()
    activation = 'relu'
    autoencoder.add(Dense(512, activation=activation, input_shape=(784,)))
    autoencoder.add(Dense(128, activation=activation))
    autoencoder.add(Dense(64, activation=activation))
    autoencoder.add(Dense(32, activation=activation))
    autoencoder.add(Dense(2, activation='linear', name="encoded"))
    autoencoder.add(Dense(32, activation=activation))
    autoencoder.add(Dense(64, activation=activation))
    autoencoder.add(Dense(128, activation=activation))
    autoencoder.add(Dense(512, activation=activation))
    autoencoder.add(Dense(784, activation=activation))
    autoencoder.compile(loss='mean_squared_error', optimizer=Adam())
    log = autoencoder.fit(x_train, x_train, batch_size=512, epochs=20, verbose=1, validation_data=(x_test, x_test))
    encoder = Model(autoencoder.input, autoencoder.get_layer('encoded').output)

    return log, encoder, autoencoder


def autoencoder(x_train, y_train, x_test, y_test):
    # Normalize the data for the autoencoder (pca needs it)
    x_train_scaled = scale(flatten_x(x_train))
    x_test_scaled = scale(flatten_x(x_test))
    log, encoder, autoencoder = train_autoencoder(x_train_scaled, x_test_scaled)
    pca_train_loss, pca_test_loss, pca = pca_loss(x_train_scaled, x_test_scaled)
    print('on train data, pca loss: %.3f, encoder loss: %.3f' % (pca_train_loss, log.history['loss'][-1]))
    print('on test data, pca loss: %.3f, encoder loss: %.3f' % (pca_test_loss, log.history['val_loss'][-1]))

    # plot
    idx = np.random.randint(len(x_train) + len(x_test), size=5000)
    y = np.hstack((y_train, y_test))[idx]
    x = np.vstack((x_train_scaled, x_test_scaled))[idx, :]
    pca_embading = pca.transform(x)
    encoder_embading = encoder.predict(x)

    f, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True)
    ax1.scatter(pca_embading[:, 0], pca_embading[:, 1], c=y)
    ax1.set_title("pca embadings")
    ax2.scatter(encoder_embading[:, 0], encoder_embading[:, 1], c=y)
    ax2.set_title("encoder embadings")
    plt.savefig("plots/autoencoder")
    plt.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    compare_models_pickle = True
    hyperparameter_pickle = True
    classes = np.unique(y_train)

    # autoencoder(x_train, y_train, x_test, y_test)

    y_train = keras_utils.to_categorical(y_train, len(classes))
    y_test = keras_utils.to_categorical(y_test, len(classes))

    # plot_decode_encode(load_model("models/encoder.pickle"), load_model("models/autoencoder.pickle"),
    #                    np.expand_dims(x_test, axis=3))
    # multi_train_multi_layer_linear_model(x_train, y_train, x_test, y_test, classes)
    # compare_models(x_train, y_train, x_test, y_test, classes, compare_models_pickle)
    hyper_parameter_covnet(x_train, y_train, x_test, y_test, classes, hyperparameter_pickle)

    # covnet_score = evalute_model(x_test, y_test, covnet_model)
    # print('covnet_score ', covnplot_modelet_score)
