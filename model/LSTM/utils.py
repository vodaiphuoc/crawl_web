import matplotlib.pyplot as plt
from typing import List
from torchmetrics.regression import MeanAbsolutePercentageError
import torch

class Report(object):
    def __init__(self, target: List[float], predict: List[float], epoch:int):
        self.metric = MeanAbsolutePercentageError()
    
        value = self.metric(torch.tensor(predict), torch.tensor(target))

        self._make_plot(target = target, predict= predict, metric_value= value.item(), epoch= epoch)

    def _make_plot(self, target: List[float], predict: List[float], metric_value: float, epoch: int):
        fig = plt.figure(figsize = (4,8))
        ax = fig.add_subplot()

        ax.plot(list(range(len(target))),target, color = 'green', marker = 'o', label = 'target price')
        ax.plot(list(range(len(target))),predict, color = 'red', marker = '+', label = 'predict price')
        ax.set_title(label= f"Price plot with MAPE: {metric_value} at epoch: {epoch}")
        legend = ax.legend(loc='upper', shadow=True, fontsize='x-large')

        fig.savefig('price_plot.png')

