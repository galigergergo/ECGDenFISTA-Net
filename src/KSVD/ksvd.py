import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def orthogonal_matching_pursuit(f, D, threshold, T_0, output):
    N = D.shape[1]       # Number of atoms in the dictionary
    coefficients = []    # List to store the coefficients (an)
    atom_indices = []    # List to store the indices of the selected atoms (γn)
    residual = f.copy()  # Initialize the residual
    approx_vector = np.zeros(N)

    # check normalization of D
    if not np.allclose(np.linalg.norm(D, axis=0), np.ones(N)):
        print('normalization is wrong', np.linalg.norm(D, axis=0))
        return -1

    # let iteration run for at most maxIter
    for n in range(T_0):
        # Calculate inner products
        inner_products = np.abs(np.dot(residual, D))

        # Find the atom with the maximum inner product and add to atom indices
        selected_index = np.argmax(inner_products)
        atom_indices.append(selected_index)

        # compute the best projection onto the current with pseudo-inverse
        # solves the least square problem
        approx_vector[atom_indices] = np.linalg.pinv(D[:, atom_indices]) @ f

        # compute the residual
        residual = f - D @ approx_vector

        if np.linalg.norm(residual) <= threshold:
            if output:
                print('reached threshold')
            break

    if np.count_nonzero(approx_vector) > T_0:
        return -1

    return coefficients, atom_indices, approx_vector


def process_column_omp(args):
    i, training_signals, dictionary, thresh, T_0, output = args
    return orthogonal_matching_pursuit(training_signals[:, i], dictionary, thresh, T_0, output)[2]


def KSVD(dictionary, training_signals, thresh,
         maxIter=100, method='omp_par', T_0=None, lambda_param = 0.1,
         thresh_zeroing=1e-6, output=False, output_plot=False):
    '''
    :param dictionary: The current dictionary to be optimized with KSVD (shape (M, K))
    :param training_signals: The training signals (shape (M, N))
    :param thresh: Threshhold to stop the iterations, when either the appoximation is good enough or when the 
                    dictionary did not change more than the norm
    :param maxIter: Maximum number of iterations
    :param method: method for pursuit algorithm. possibilities are ['omp', 'omp_par', 'basis', 'bpnd']
    :param T_0: sparseness for the pursuit algorithm
    :param lambda_param: lambda parameter for the bpdn algorithm
    :param thresh_zeroing:
    :param output: whether to print the 
    :return:
    '''
    dim_2, K = dictionary.shape
    dim, N = training_signals.shape
    assert dim == dim_2
    residual_mp = []
    residual_svd = []

    # normalize columns in l^2 norm
    dictionary /= np.linalg.norm(dictionary, axis=0)

    for iteration in tqdm(range(maxIter)):
        # for later comparison
        dictionary_old = dictionary. copy()
        # sparse coding stage, using method as in method option
        if method == 'omp':
            X = np.zeros((K, N))
            for i in range(N):
                X[:, i] = orthogonal_matching_pursuit(training_signals[:, i], dictionary,
                                                      threshold=thresh, T_0=T_0, output=output)[2]

        elif method == 'omp_par':
            with ProcessPoolExecutor(max_workers=4) as executor:
                args = [(i, training_signals, dictionary, thresh, T_0, output) for i in range(N)]
                results = list(executor.map(process_column_omp, args))
            X = np.column_stack(results)

        elif method == 'basis':
            X = np.zeros((K, N))
            for i in range(N):
                X[:, i] = basis_pursuit(training_signals[:, i], dictionary)

        elif method == 'bpdn':
            X = np.zeros((K, N))
            # sparse coding stage
            for i in tqdm(range(N)):
                alpha = BPDN(training_signals[:, i], dictionary, lambda_param=lambda_param)[1]
                alpha[np.abs(alpha) < thresh_zeroing] = 0.0
                X[:, i] = alpha
        else:
            print('no such method, method must be one of [\'omp\', \'omp_par\', \'basis\', \'bpdn\'] ')
            return -1

        if output:
            print(method, f'iteration = {iteration}, non-zero = {np.count_nonzero(X)} out of {X.shape}, ' +\
                          f'residual = {np.linalg.norm(training_signals - dictionary @ X)}')

        residual_mp.append(np.linalg.norm(training_signals - dictionary @ X))

        # codebook update stage
        for k in range(K):
            omega_vector = np.nonzero(X[k, :])[0]
            if len(omega_vector) == 0:
                # no nonzero element exist
                max_idx = np.argmax(np.linalg.norm(training_signals - dictionary @ X, axis=0))
                X[:, max_idx] = 0
                X[k, max_idx] = np.linalg.norm(training_signals[:, max_idx])
                dictionary[:, k] = training_signals[:, max_idx] / np.linalg.norm(training_signals[:, max_idx])
                continue

            # compute E_k and only consider relevant rows by multiplication with Omega
            Omega = np.eye(N)[:, omega_vector]
            E_k = training_signals - np.sum(dictionary[:, np.arange(K)!=k, np.newaxis] * X[np.newaxis, np.arange(K)!=k, :], axis=1)
            ER_k = E_k@Omega

            # compute singular value decomposition
            U, S, Vh = np.linalg.svd(ER_k)

            # use first collumn of U and V*S to update codebook
            dictionary[:, k] = U[:, 0]
            X[k, omega_vector] = Vh[0, :] * S[0]

        # set almost zero values to zero
        X[np.abs(X) < 1e-10] = 0.0 # probably does not do anything

        if output:
            print(f'iteration = {iteration}, non-zero = {np.count_nonzero(X)}, ' +\
                  f'residual = {np.linalg.norm(training_signals - dictionary@X)}')
            residual_svd.append(np.linalg.norm(training_signals - dictionary@X))

        if np.linalg.norm(training_signals - dictionary@X) < thresh:
            print('stopped from res norm')
            break
        if np.linalg.norm(dictionary_old - dictionary) < thresh:
            print('stopped from dictionary changing', np.linalg.norm(dictionary_old - dictionary))
            break

    return dictionary, X