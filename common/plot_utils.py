import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_attn_weights(attn_weights):
    """
    attn_weights: (N, S, S)
    """
    plt.clf()
    N, S, _ = attn_weights.shape
    attn_weights = np.mean(attn_weights, 0)
    df_cm = pd.DataFrame(attn_weights, range(S), range(S))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.2)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size
    #wandb.log({"attn":plt})
    plt.show()
    plt.savefig(os.path.join(wandb.run.dir, 'attn_weights.png'),dpi=200)
    wandb.save('*png')