from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adadelta
from keras.losses import categorical_crossentropy
from keras import utils as keras_utils
import numpy as np
from matplotlib import pyplot as plt
import pickle


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
    for log in model_logs:
        if only_val == False:
            plt.plot(log.history['acc'])
        plt.plot(log.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(legend)
    plt.title(title)

    plt.subplot(2, 1, 2)
    for log in model_logs:
        if only_val == False:
            plt.plot(log.history['loss'])
        plt.plot(log.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legend)
    # plt.title(title + " loss")
    plt.tight_layout()
    plt.show()


def plot_model(model_log, model_name):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(model_log.history['acc'])
    plt.plot(model_log.history['val_acc'])
    plt.title(model_name + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])

    plt.subplot(2, 1, 2)
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title(model_name + 'loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    plt.show()


def train_model(x_train, y_train, x_test, y_test, model, opt=SGD()):
    model.compile(optimizer=opt,
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

    model_log = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2,
                          validation_data=(x_test, y_test))

    # plot_model(model_log, model_name)
    return model, model_log


def flatten_x(X):
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def train_multi_layer_linear_model(x_train, y_train, x_test, y_test, num_classes):
    x_train = flatten_x(x_train)
    x_test = flatten_x(x_test)
    model = Sequential([
        Dense(256, input_shape=(x_train.shape[1],)),
        Activation('softmax'),
        # Dense(64),
        # Activation('softmax'),
        Dense(256),
        Activation('softmax'),
        Dense(num_classes),
        Activation('softmax'),
    ])

    return train_model(x_train, y_train, x_test, y_test, model)


def train_linear_model(x_train, y_train, x_test, y_test, num_classes):
    x_train = flatten_x(x_train)
    x_test = flatten_x(x_test)
    model = Sequential([
        Dense(256, input_shape=(x_train.shape[1],)),
        Activation('softmax'),
        Dense(num_classes),
        Activation('softmax'),
    ])

    return train_model(x_train, y_train, x_test, y_test, model)


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
    if from_pickle:
        models = pickle.load(open("models.pickle", "rb"))
        linear_model, linear_log = (models['linear']['model'], models['linear']['log'])
        mlp_model, mlp_log = (models['mlp']['model'], models['mlp']['log'])
        covnet_model, covnet_log = (models['covnet']['model'], models['covnet']['log'])
    else:
        # linear_model, linear_log = train_linear_model(x_train, y_train, x_test, y_test, len(classes))
        # mlp_model, mlp_log = train_multi_layer_linear_model(x_train, y_train, x_test, y_test, len(classes))
        covnet_model, covnet_log = train_covnet(x_train, y_train, x_test, y_test, len(classes), opt=SGD(0.0001))

        pickle.dump({'linear': {'log': linear_log, 'model': linear_model},
                     'mlp': {'log': mlp_log, 'model': mlp_log},
                     'covnet': {'log': covnet_log, 'model': covnet_model}
                     }, open("models.pickle", "wb"))

    plot_models([linear_log, mlp_log, covnet_log], ['linear', 'mlp', 'covnet'], 'Compare between models', False)


def hyper_parameter_covnet(x_train, y_train, x_test, y_test, classes, from_pickle=True):
    if from_pickle:
        models = pickle.load(open("covnet_hyperparameter.pickle", "rb"))
        lr1_model, lr1_log = (models['lr1']['model'], models['lr1']['log'])
        lr01_model, lr01_log = (models['lr01']['model'], models['lr01']['log'])
        lr001_model, lr001_log = (models['lr001']['model'], models['lr001']['log'])
        lr0001_model, lr0001_log = (models['lr0001']['model'], models['lr0001']['log'])

    else:
        lr1_model, lr1_log = train_covnet(x_train, y_train, x_test, y_test, len(classes), opt=SGD(0.1))
        lr01_model, lr01_log = train_covnet(x_train, y_train, x_test, y_test, len(classes), opt=SGD(0.01))
        lr001_model, lr001_log = train_covnet(x_train, y_train, x_test, y_test, len(classes), opt=SGD(0.001))
        lr0001_model, lr0001_log = train_covnet(x_train, y_train, x_test, y_test, len(classes), opt=SGD(0.0001))

        pickle.dump({'lr1': {'log': lr1_log, 'model': lr1_model},
                     'lr01': {'log': lr01_log, 'model': lr01_model},
                     'lr001': {'log': lr001_log, 'model': lr001_model},
                     'lr0001': {'log': lr0001_log, 'model': lr0001_model}
                     }, open("models.pickle", "wb"))

    plot_models([lr1_log, lr01_log, lr001_log, lr0001_log], ['0.1', '0.01', '0.001', '0.0001'],
                'Covnet hyperparameter - learning rate', True)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    compare_models_pickle = False
    hyperparameter_pickle = False
    classes = np.unique(y_train)
    y_train = keras_utils.to_categorical(y_train, len(classes))
    y_test = keras_utils.to_categorical(y_test, len(classes))

    # x_train = x_train[:1000]
    # y_train = y_train[:1000]
    # compare_models(x_train, y_train, x_test, y_test, classes, compare_models_pickle)
    hyper_parameter_covnet(x_train, y_train, x_test, y_test, classes, hyperparameter_pickle)

    # covnet_score = evalute_model(x_test, y_test, covnet_model)
    # print('covnet_score ', covnplot_modelet_score)
