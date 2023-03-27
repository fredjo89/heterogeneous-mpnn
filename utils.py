def get_num_params(model):
    """
    Returns the number of parameters of a model
    """

    total_dim = 0
    for param in model.parameters():
        my_shape = param.data.shape
        dim = 1
        for i in range(len(my_shape)):
            dim = dim * max(my_shape[i], 1)
        total_dim += dim
    return total_dim