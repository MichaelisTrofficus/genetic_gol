import numpy as np
import matplotlib.pyplot as plt


def visualize_board(board_array):
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    axes.imshow(board_array, cmap="Greys")
    axes.grid(axis='both', linestyle='-', color='k', linewidth=1)
    axes.set_xticks(np.arange(0.5, 25.5, 1))
    axes.set_yticks(np.arange(0.5, 25.5, 1))
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.tick_params(axis='both', which='both', length=0)
    plt.show()


def visualize_boards(board_array1, board_array2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    axes[0].imshow(board_array1, cmap="Greys")
    axes[0].grid(axis='both', linestyle='-', color='k', linewidth=1)
    axes[0].set_xticks(np.arange(0.5, 25.5, 1))
    axes[0].set_yticks(np.arange(0.5, 25.5, 1))
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].tick_params(axis='both', which='both', length=0)

    axes[1].imshow(board_array2, cmap="Greys")
    axes[1].grid(axis='both', linestyle='-', color='k', linewidth=1)
    axes[1].set_xticks(np.arange(0.5, 25.5, 1))
    axes[1].set_yticks(np.arange(0.5, 25.5, 1))
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis='both', which='both', length=0)

    plt.show()


def visualize_evolution(df, experiment_id):
    """
    Representa en la mismo plot la distribución de células
    al inicio y al final del delta temporal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    experiment_row = df.loc[experiment_id, :].values
    delta = experiment_row[1]
    start_board = np.reshape(experiment_row[2: 2+625], (25, 25))
    stop_board = np.reshape(experiment_row[625+2:], (25, 25))

    print(f"Experiment id: {experiment_id}")
    print(f"Delta: {delta}")

    axes[0].imshow(start_board, cmap="Greys")
    axes[0].grid(axis='both', linestyle='-', color='k', linewidth=1)
    axes[0].set_xticks(np.arange(0.5, 25.5, 1))
    axes[0].set_yticks(np.arange(0.5, 25.5, 1))
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].tick_params(axis='both', which='both', length=0)

    axes[1].imshow(stop_board, cmap="Greys")
    axes[1].grid(axis='both', linestyle='-', color='k', linewidth=1)
    axes[1].set_xticks(np.arange(0.5, 25.5, 1))
    axes[1].set_yticks(np.arange(0.5, 25.5, 1))
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis='both', which='both', length=0)

    fig.suptitle(f"Experiment id: {experiment_id} | Delta: {delta}")
    plt.show()
