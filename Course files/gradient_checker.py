import numpy as np

def compute_numerical_gradient(generic_function, variable, h=1e-5):
    """
    Computes for the numerical gradient of the `generic_function` with respect
    to `variable`.
    
    Inputs:
    - generic_function: A function that takes in `variable` as its input and outputs
        a scalar value.
    - variable: A vector / matrix of arbitrary size / dimensions. We compute
        gradients of the `generic_function` with respect to this `variable.

    
    Returns:
    - grad: A vector / matrix corresponding to the gradients of the 
        `generic_function` with respect to `variable`. It should have 
        the same dimensions as `variable`.
    """

    # Computes for the output using the current values of variable
    output = generic_function(variable)

    grad = np.zeros_like(variable)

    # A way to iterate through the elements of a numpy array / matrix
    # See the links below for more details:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.nditer.html
    # https://docs.scipy.org/doc/numpy/reference/arrays.nditer.html#arrays-nditer

    it = np.nditer(variable, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # Gets the index of the current element 
        idx = it.multi_index

        #############################################################################
        # TODO: Create a copy of the original value of current element of           #
        # `variable`.                                                               #
        #############################################################################
        var_copy = variable[idx]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        #############################################################################
        # TODO: Compute for the numerical gradient of the current element using     #
        # the central difference formula.                                           #
        #############################################################################
        variable[idx] = var_copy + h
        loss_h_plus = generic_function(variable)

        variable[idx] = var_copy - h
        loss_h_minus = generic_function(variable)

        grad[idx] = (loss_h_plus - loss_h_minus) / (2 * h)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        #############################################################################
        # TODO: Restore the current elemend of `variable` to its original value.    #
        #############################################################################
        variable[idx] = var_copy
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        it.iternext()
    return grad

def relative_error(x,y):
    """
    Computes for the relative error between two vectors / matrices of the same
    size.
    
    Inputs:
    - x: A vector / matrix of arbitrary size / dimensions.
    - y: A vector / matrix of arbitrary size / dimensions.

    
    Returns:
    - rel_error: A scalar value representing the maximum relative error between
        x and y.
    """
    rel_error = np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
    return rel_error

