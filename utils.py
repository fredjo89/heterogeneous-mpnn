import math, random, torch, matplotlib.pyplot as plt, pandas as pd, logging
from sklearn.metrics import roc_curve, roc_auc_score


def get_num_params_of_model(model):
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


def plot_training_hist(train_hist):
    """
    DocString...
    """
    fig, ax = plt.subplots(figsize=(5, 2.7))
    ax.plot(train_hist["loss_train"], label="train")
    ax.plot(train_hist["loss_val"], label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.legend()
    plt.show()


def _draw(bool_array, draw_frac):
    """
    A helper function that draws indices from a boolean array where the elements are true,
    and returns an array of the same dimension where only the drawn indices are true

    Parameters
    ----------
    bool_array (torch.Tensor): An array of boolean values used to indicate which elements should be drawn from.
    draw_frac (float):  Fraction of values to draw

    Returns
    -------
    bool_array (torch.Tensor): An array of boolean values with the drawn samples.

    Example
    -------
    bool_array = [True True False True True]
    draw_frac = 0.50
    bool_array_drawn = [False True False True False]
    """
    # Get array of indices where "bool_array == True"
    idx_true = bool_array.nonzero()[:, 0]
    # From the indices, we will draw "draw_frac" of them:
    num_draws = math.ceil(idx_true.shape[0] * draw_frac)
    idx_true_draws = idx_true[random.sample(range(0, idx_true.shape[0]), num_draws)]
    # We create a new boolean array which is True only at the drawn indices
    bool_array_drawn = torch.zeros(bool_array.shape[0], dtype=torch.bool)
    bool_array_drawn[idx_true_draws] = True
    return bool_array_drawn


def train_val_test_split(val_frac, test_frac, y):
    """
    Creates train/validation/test masks.
    Stratified random sampling is employed to create train/validation/test masks with
    an allocation that is proportionate to the original class balance, thereby ensuring
    that the class balance is maintained across all three sets.

    Parameters
    ----------
    val_frac (float): Fraction of observations that will be assigned to the validation set
    test_frac (float): Fraction of observations that will be assigned to the test set
    y (torch.Tensor): The array that contains the labels of each observation

    Returns
    -------
    train_mask (torch.Tensor): Train mask
    val_mask (torch.Tensor): Validation mask
    test_mask (torch.Tensor): Test Mask

    """
    pos_mask = y == 1
    neg_mask = y == 0

    ## Creating test-mask ##
    pos_test = _draw(pos_mask, test_frac)  # Drawing observations for y == 1
    neg_test = _draw(neg_mask, test_frac)  # Drawing observations for y == 0
    test_mask = (pos_test) | (neg_test)
    ## Creating val-mask ##
    pos_val = _draw(
        pos_mask & (~test_mask), val_frac / (1 - test_frac)
    )  # Drawing observations for y == 1
    neg_val = _draw(
        neg_mask & (~test_mask), val_frac / (1 - test_frac)
    )  # Drawing observations for y == 0
    val_mask = (pos_val) | (neg_val)
    ## Creating train-mask ##
    train_mask = (pos_mask | neg_mask) & (~test_mask) & (~val_mask)
    return train_mask, val_mask, test_mask


def _early_stopping(train_hist):
    """
    DocString...
    """
    epoch = train_hist.shape[0] - 1
    if train_hist.loc[epoch, "loss_val"] > train_hist.loc[epoch - 1, "loss_val"]:
        return True

    return False


def train_model(
    model,
    data,
    node_type_to_classify,
    learning_rate=1e-2,
    weight_decay=1e-10,
    min_epochs=1,
    max_epochs=100,
    print_learning_progress_freq=25,
):
    """
    DocString...
    """

    # Initialize the loss-function and optimizer
    loss_fun = torch.nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-7)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    y = data[node_type_to_classify].y

    # Training model with early stopping
    train_hist = pd.DataFrame(columns=["loss_train", "loss_val"])
    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()

        pred = model.forward(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

        loss_train = loss_fun(
            pred[data[node_type_to_classify].train_mask],
            y[data[node_type_to_classify].train_mask],
        )

        loss_val = loss_fun(
            pred[data[node_type_to_classify].val_mask],
            y[data[node_type_to_classify].val_mask],
        )

        train_hist.loc[epoch] = loss_train.item(), loss_val.item()

        if train_hist.loc[epoch, "loss_val"] == train_hist["loss_val"].min():
            best_model = model

        if epoch >= min_epochs and _early_stopping(train_hist):
            logging.info(
                f"Early stopping at epoch #{epoch} (validation loss is decreasing)."
            )
            break

        if epoch % print_learning_progress_freq == 0:
            logging.info(f"Epoch #{epoch}. Validation loss: {loss_val:.4f}")

        loss_train.backward()
        optimizer.step()

    return best_model, train_hist


def _plot_roc_curve(pred, y, mask, ax, split):
    """
    DocString...
    """

    pred = pred.detach().numpy()
    y = y.detach().numpy()
    mask = mask.detach().numpy()

    pred_split = pred[mask]
    y_split = y[mask]
    fpr, tpr, _ = roc_curve(y_split, pred_split)
    label = f"{split} AUC ({roc_auc_score(y_split, pred_split):.3f})"
    ax.plot(fpr, tpr, marker=".", markersize=2, linewidth=1, label=label)


def plot_roc_curves(data, pred, node_type_to_classify):
    """
    DocString...
    """
    y = data[node_type_to_classify].y
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_roc_curve(pred, y, data[node_type_to_classify].train_mask, ax, "Train")
    _plot_roc_curve(pred, y, data[node_type_to_classify].val_mask, ax, "Val")
    _plot_roc_curve(pred, y, data[node_type_to_classify].test_mask, ax, "Test")
    ax.plot([0, 1], [0, 1], linestyle="--", color="b")
    ax.legend()
    plt.show()
