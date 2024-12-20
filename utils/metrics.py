import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error


def compute_metrics(output,label):
    label = label.unsqueeze(1)
    output = output.to('cpu')
    label = label.to('cpu')
    output = np.array(output)
    label = np.array(label)
    MAE = mean_absolute_error(label, output)
    SRC, _ = spearmanr(output, label)
    nMSE = np.mean(np.square(output - label)) / (label.std() ** 2)
    PCC = pearsonr(output.squeeze(-1), label.squeeze(-1))[0]
    return MAE, SRC, nMSE, PCC