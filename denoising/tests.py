from Image_Denoising import *


def learn_MVN_test():
    # set a 2X3 matrix, and make sure the learned mean and cov are correct
    X = np.array([[1,1,1], [2,2,2]])
    mean = np.array([1,2])
    cov = np.array([[0,0], [0,0]])
    mvn_model = learn_MVN(X)
    print(mvn_model.cov)
    assert ((mean == mvn_model.mean).all())
    assert ((cov == mvn_model.cov).all())
    print("pass learn MVN")


def denoise_function_mock(patch, model, std):
    print(patch.shape)


def denoise_one_image():
    path = "test_images.pickle"
    with open(path, 'rb') as f:
        train_pictures = pickle.load(f)

    pic = grayscale_and_standardize(train_pictures)[0]

    denoise_image(pic, None, denoise_function_mock, 1)

if __name__ == '__main__':
    # denoise_one_image()
    learn_MVN_test()