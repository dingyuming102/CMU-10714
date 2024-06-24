#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matrix_multiplication(const float *X, const float *Y, float *Z, int m, int n, int k) {
    // X: m x n, Y: n x k, Z: m x k
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) 
        {
            Z[i * k + j] = 0;
            for (int s = 0; s < n; s++)
                Z[i * k + j] += X[i * n + s] * Y[s * k + j];
        }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    
    // num_examples = X.shape[0]
    // iteration_num = math.ceil(num_examples / batch)
    // for i in range(iteration_num):
    //     start_idx, end_idx = i * batch, min((i+1)*batch, X.shape[0])  # Ensure the index does not go out of bounds
    //     m = end_idx - start_idx
        
    //     X_b = X[start_idx: end_idx, :]
    //     y_b = y[start_idx: end_idx]
        
    //     Z = np.exp(X_b @ theta)
    //     Z /= np.sum(Z, axis=1, keepdims=True)
    //     Iy = np.zeros_like(Z)
    //     Iy[np.arange(m), y_b] = 1
        
    //     grad = X_b.T @ (Z - Iy) / m
    //     assert(grad.shape == theta.shape)
    //     theta -= lr * grad

    // X: m x n, y: 1 x m, theta: n x k
    int iterations = (m + batch - 1) / batch;
    for (int iter = 0; iter < iterations; iter++) {
        const float *X_b = &X[iter * batch * n]; // x: batch x n
        float *Z = new float[batch * k];     // Z: batch x k
        matrix_multiplication(X_b, theta, Z, batch, n, k);
        for (int i = 0; i < batch * k; i++) 
            Z[i] = exp(Z[i]); // element-wise exp
        for (int i = 0; i < batch; i++) {
            float sum = 0;
            for (int j = 0; j < k; j++) sum += Z[i * k + j];
            for (int j = 0; j < k; j++) Z[i * k + j] /= sum; // row-wise normalization
        }
        for (int i = 0; i < batch; i++) 
            Z[i * k + y[iter * batch + i]] -= 1; // minus one-hot vector
        float *X_b_T = new float[n * batch];
        float *grad = new float[n * k];
        for (int i = 0; i < batch; i++) 
            for (int j = 0; j < n; j++) 
                X_b_T[j * batch + i] = X_b[i * n + j];
        matrix_multiplication(X_b_T, Z, grad, n, batch, k);
        for (int i = 0; i < n * k; i++) theta[i] -= lr / batch * grad[i]; // SGD update
        delete[] Z;
        delete[] X_b_T;
        delete[] grad;
    }


    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
