import pickle

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
from skimage.util import view_as_windows as viewW

EM_MAX_ITERATIONS = 100
EM_STOP_EPSILION = 0.0001

def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open(path, 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                                                                          window[0] * window[1]).T[
           :, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    # gray_image = grayscale_and_standardize([image])[0]
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(denoise_image(noisy_images[:, :, i], model, denoise_function,
                                             noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    mean - k-length mean vector for the each of the gaussians
    """

    def __init__(self, cov, mix, mean = None):
        self.cov = cov
        self.mix = mix
        self.mean = np.array([0] * cov.shape[1])


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """

    def __init__(self, P, vars, mix):
        self.P = P
        self.vars = vars
        self.mix = mix


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """

    return np.sum((multivariate_normal.logpdf(X.T, mean=model.mean, cov=model.cov,
                                              allow_singular=False)))


# def GSM_log_likelihood(X, model):
#     """
#     Given image patches and a GSM model, return the log likelihood of the patches
#     according to the model.
#
#     :param X: a patch_sizeXnumber_of_patches matrix of image patches.
#     :param model: A GSM_Model object.
#     :return: The log likelihood of all the patches combined.
#     """
#     N = X.shape[0]
#     k = model.mix.shape[0]
#     inners = np.zeros((N, k))
#     c_upper = np.ones((N, k))
#     for kk in range(k):
#         inners[:, kk] = np.log(model.mix[kk]) + multivariate_normal.logpdf(X, model.mean,
#                                                                            model.cov[kk], allow_singular=True)
#         c_upper[:, kk] = np.log(model.mix[kk]) + multivariate_normal.logpdf(X, model.mean, model.cov[kk], allow_singular=True)
#
#     c_lower = logsumexp(c_upper, axis=1).reshape(N, 1)
#     c = c_upper - c_lower
#
#     for kk in range(k):
#         inners[:, kk] += c[:, kk]
#
#     return 10000 + np.sum(inners)

def keren_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    N = X.shape[0]
    k = model.mix.shape[0]
    inners = np.zeros((N, k))
    for kk in range(k):
        inners[:, kk] = np.log(model.mix[kk]) + multivariate_normal.logpdf(X, np.zeros(64),
                                                                           model.cov[kk],
                                                                           allow_singular=True)
    c_lower = logsumexp(inners, axis=1).reshape(N, 1)
    c = inners - c_lower

    for kk in range(k):
        inners[:, kk] += c[:, kk]

    return 1000000 + np.sum(inners)

def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """

    k = len(model.mix)
    # The likelihood for each patch
    likelihood = np.zeros((X.shape[1], k))
    for i in range(k):
        likelihood[:, i] = np.log(model.mix[i]) + multivariate_normal.logpdf(X.T, mean=model.mean,
                                                                       cov=model.cov[i])

    return 1000000 + np.sum(logsumexp(likelihood, axis=1))

def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    # TODO: YOUR CODE HERE


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    mean = np.mean(X, 1)
    cov = np.cov(X)
    model = MVN_Model(mean, cov)
    return model


def posterior_prob(patchs, pi, mean, cov_vec):
    '''
    :param patchs:
    :param pi: k length vector, each cell i contains the probability of a patch to be drawn from
    gaussian i
    :param mean: scalar, assume mean is the same for all gaussians
    :param cov_vec: k length vector where each cell is a cov matrix for the i'th gaussian
    :return: kXM matrix, i,j cell: probability of patch i to be from gaussian j
    '''

    k = len(cov_vec)
    c = np.ones((patchs.shape[1], len(cov_vec)))
    for kk in range(k):
        # calculate c: calculate numerator and denominator separately in logspace,
        # and then divide and revert to original space
        c[:, kk] = np.log(pi[kk]) + multivariate_normal.logpdf(patchs.T, mean, cov_vec[kk],
                                                               allow_singular=True)

    denominator_log = logsumexp(c, axis=1).reshape(patchs.shape[1], 1)
    c = np.exp(c - denominator_log)
    return c
    c = np.ones((N, k))  # probability of y given a guassian distribution, N x k
    k = len(cov_vec)
    log_pi = np.log(pi)

    log_PDF_matrix = np.zeros((patchs.shape[1], k))
    # PDF_matrix = np.zeros((patchs.shape[1], k))
    # numerator = np.zeros((patchs.shape[1], k))
    # denominator = np.zeros(patchs.shape[1])
    for i in range(k):
        # E step - we do it in log space and then take the exponent
        log_PDF_matrix[:, i] = multivariate_normal.logpdf(patchs.T, cov=cov_vec[i], mean=mean)
    #     PDF_matrix[:, i] = multivariate_normal.pdf(patchs.T, cov=cov_vec[i], mean=mean)
    #     numerator[:, i] = PDF_matrix[:, i] * pi[i]
    #     denominator += numerator[:, i]
    #
    # c = numerator - np.matlib.repeat(denominator[:, None], k, 1)
    # return c
    log_numerator = log_PDF_matrix + log_pi
    log_denominator = logsumexp(log_PDF_matrix + log_pi, axis=1)
    c = np.exp(log_numerator - np.matlib.repeat(log_denominator[:, None], k, 1))
    return c


def Expectation_Maximization(samples, k, model, max_iterations=100, learn_gsm=False):
    '''
    Run the iterative EM Algorithm on the given data.
    :param samples: [numpy.ndarray] 2d array, each row contains a flattened image patch of size d. Nxd
    :param k: number of guassians
    :param max_iterations: stop at this number of iterations, if not yet converged
    :param learn_gsm: if True, do not learn mean and assume cov=r*single_cov, where we need to learn r.
    :return: mean, covariance, pi, list of log likelihoods for each iteration
    '''
    N, d = samples.shape

    # initialize parameters
    c = np.ones((N, k))  # probability of y given a guassian distribution, N x k
    pi = model.mix  # mixture, probability of y, multiplication of c, 1 x k
    covariance = model.cov  # k x d x d
    r_squared = np.random.rand(k,1)
    if learn_gsm:
        mean = np.array([0] * d)
    else:
        mean = model.means  # k x d

    # for gsm learning
    if learn_gsm:
        cov_tmp = samples - mean[0]
        single_cov = cov_tmp.T.dot(cov_tmp) / N
        base_cov = np.array([single_cov] * k)
        covariance = base_cov

    # convergence parameter
    epsilon = 0.0001

    loglikelihoods = []
    iters = 0

    while iters < max_iterations:
        print("iteration:", iters)
        for kk in range(k):
            # calculate c: calculate numerator and denominator separately in logspace,
            # and then divide and revert to original space
            c[:, kk] = np.log(pi[kk]) + multivariate_normal.logpdf(samples, mean,
                                                                   covariance[kk], allow_singular=True)

        denominator_log = logsumexp(c, axis=1).reshape(N, 1)
        c = np.exp(c - denominator_log)

        # calculate pi
        pi = np.sum(c, axis=0) / N

        # calculate mu (mean)
        if not learn_gsm:
            for kk in range(k):
                mean[kk] = np.dot(c[:, kk].T, samples) / np.sum(c[:, kk])

        # calculate r
        if learn_gsm:
            for kk in range(k):
                numerator = np.sum(c[:, kk] * np.diag(samples.dot(np.linalg.pinv(base_cov[kk])).dot(samples.T)))
                r_squared[kk] = numerator / (d * np.sum(c[:, kk]))

        # calculate Sigma (covariance)
        for kk in range(k):
            if learn_gsm:
                covariance[kk] = base_cov[kk] * r_squared[kk]
            else:
                covariance[kk] = np.dot(c[:, kk] * (samples - mean[kk]).T, (samples - mean[kk]))
                covariance[kk] = covariance[kk] / np.sum(c[:, kk])

        # calculate log likelihood of parameters
        tmp_model = GSM_Model(covariance, pi)
        loglikelihoods.append(keren_log_likelihood(samples, tmp_model))
        print("likelihood:", loglikelihoods[-1])
        if iters >= 2 and np.abs(loglikelihoods[iters] - loglikelihoods[iters-1]) < epsilon:
            break
        iters += 1

    # plot_lle(loglikelihoods)
    return mean, covariance, pi, loglikelihoods


def keren_GSM(X, k):
    X = X.transpose()

    #initialize model
    mean = np.zeros((k, 64))
    cov_tmp = X - mean[0]
    single_cov = cov_tmp.T.dot(cov_tmp) / X.shape[0]
    base_cov = np.array([single_cov] * k)
    for kk in range(k):
        base_cov[kk] /= (kk+1)
    mix = np.array([float(1 / k)] * k)
    # gmm_model = GMM_Model(mix, mean, base_cov)
    initial_model = GSM_Model(base_cov, mix)

    learnt_mean, learnt_cov, learnt_mix, lle = Expectation_Maximization(X, k, initial_model,
                                                                        max_iterations=4,
                                                                        learn_gsm=True)
    learnt_model = GSM_Model(learnt_cov, learnt_mix)
    return learnt_model

def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """

    # the probability x_i came from gaussian y, we start with uniform
    pi = np.array([1 / k] * k)

    # start with random scalars
    # TODO: maybe better to do diffrenet initialization
    r_y = np.array(np.random.normal(size=k))
    d = X.shape[0]
    mean = np.zeros(d)

    cov = np.cov(X)
    # It's a k length vector, where each cell is the scaled cov matrix
    # cov_vec = np.zeros((k, d, d))
    # cov_vec = np.array([(y+1) * cov for y in range(k)])
    # r_y = np.array(np.random.normal(size=k))
    r_y = np.array(np.random.rand(k)) * 10
    cov_vec = np.array([r_y[i] * cov for i in range(k)])
    assert(cov_vec[0, :, :].shape == cov.shape)

    likelihood_results = []

    numeator_mul = np.diag(X.T.dot(np.linalg.pinv(cov)).dot(X))
    r_y = np.zeros(k)
    for j in range(1, EM_MAX_ITERATIONS):
        print("iteration:", j)
        # E step - update patch probabilty for each gaussian
        c = posterior_prob(X, pi, mean, cov_vec)
        # k = len(cov_vec)
        # log_pi = np.log(pi)
        # log_PDF_matrix = np.zeros((X.shape[1], k))
        # for i in range(k):
        #     # E step - we do it in log space and then take the exponent
        #     log_PDF_matrix[:, i] = multivariate_normal.logpdf(X.T, cov=cov_vec[i], mean=mean)
        #
        # log_numerator = log_PDF_matrix + log_pi
        # log_denominator = logsumexp(log_PDF_matrix + log_pi, axis=1)
        # c = np.exp(log_numerator - np.matlib.repeat(log_denominator[:, None], k, 1))
        #
        # M step - update pi
        pi_y = np.sum(c, axis=0) / X.shape[1]
        # test the probabilities are still valid
        assert (np.abs(sum(pi_y) - 1) < 0.05)
        assert ((np.abs(np.sum(c, axis=1) - 1) < 0.05).all())


        # numeator_mul = np.diag(X.T.dot(np.linalg.pinv(cov)).dot(X))
        for i in range(k):
            numerator = np.sum(c[:, i] * numeator_mul)
            r_y[i] = numerator / (d * np.sum(c[:, i]))
            if np.isnan(r_y).any():
                print(np.sum(c, axis=1))

        # numeator_mul = X.T @ (np.linalg.pinv(cov) @ X)
        # r_y = np.sum(c @ numeator_mul, axis=1) / d * np.sum(c, axis=1)
        print("r_y:", r_y)
        # Update the cov matrix
        # TODO: Can remove the loop?
        cov_vec = np.array([r_y[i] * cov for i in range(k)])
        # cov_vec = np.outer(r_y, cov).reshape((k,d,d))
        likelihood_results.append(GSM_log_likelihood(X, GSM_Model(cov_vec, pi_y, mean)))
        print('likelihood_results:', likelihood_results[-1])
        # Add condition on likelihood

        if j > 2 and np.abs(likelihood_results[-1] - likelihood_results[-2]) < EM_STOP_EPSILION:
            return GSM_Model(cov_vec, pi_y, mean)
    print("likelihood:", likelihood_results)
    return GSM_Model(cov_vec, pi_y, mean)


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """

    # TODO: YOUR CODE HERE


def weiner(Y, mu, cov, noise_std):
    noise_var = np.square(noise_std)
    cov_inv = np.linalg.inv(cov)
    normalized_y = (Y / noise_var)
    normalized_mean = cov_inv.dot(mu) #np.matmul(cov_inv, mu)

    first_mat = np.linalg.inv(cov_inv +
                              (np.eye(cov_inv.shape[0], cov_inv.shape[1]) / noise_var))
    second_mat = (normalized_mean + normalized_y.transpose()).transpose()

    return first_mat.dot(second_mat) #np.matmul(first_mat, second_mat)


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    return weiner(Y, mvn_model.mean, mvn_model.cov, noise_std)


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """

    k = len(gsm_model.cov)
    d = Y.shape[0]
    # mean = np.array([0] * d)
    mean = gsm_model.mean
    weniner_vec = np.zeros((k, d, Y.shape[1]))

    # print(Y.shape, gsm_model.mix.shape,len(mean), gsm_model.cov.shape)
    c = posterior_prob(Y, gsm_model.mix, mean, gsm_model.cov +
                       (noise_std * np.array([np.eye(gsm_model.cov.shape[1], gsm_model.cov.shape[
                           2])
                                              for _
                                              in range(3)])))



    for i in range(k):
        weniner_vec[i] = c[:, i] * weiner(Y, mean, gsm_model.cov[i], noise_std)

    return np.sum(weniner_vec, axis=0)


# def MLE_multidimensional_gaussian(x, mu, cov):
#     d = cov.shape[0]
#     exp = np.exp(-0.5 * (x - mu).transpose() * cov * (x - mu))
#     norm_factor = 1 / np.sqrt(np.power((2 * np.pi), d) * np.linalg.det(cov))
#     return norm_factor * exp


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    # TODO: YOUR CODE HERE


def learn(learn_func, path='train_images.pickle', k=None, patch_size=(8, 8)):
    with open(path, 'rb') as f:
        train_pictures = pickle.load(f)

    # gray_train_pictures = grayscale_and_standardize(train_pictures)
    patches = sample_patches(train_pictures, psize=patch_size, n=20000)
    # patches = np.array([np.random.normal(0, 1, 30), np.random.normal(0,2,30)])
    if k:
        return learn_func(patches, k)
    return learn_func(patches)


if __name__ == '__main__':

    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)
    test_pictures = grayscale_and_standardize(test_pictures)
    # model = learn(learn_MVN)
    # denoise = MVN_Denoise

    import os
    if not os.path.exists("GSM_model.pickle"):
        model = learn(learn_GSM, k=3)
        # model = learn(learn_GSM, k=3)
        with open("GSM_model.pickle", "wb") as f:
            pickle.dump(model, f)
    else:
        with open("GSM_model.pickle", "rb") as f:
            model = pickle.load(f)
    denoise = GSM_Denoise

    pic = np.random.choice(test_pictures)
    # pic = np.random
    test_denoising(pic, model, denoise, noise_range=([0.01, 0.2]))

    # pics = grayscale_and_standardize(test_pictures)
    # for pic in pics:
    #     test_denoising(pic, model, denoise)