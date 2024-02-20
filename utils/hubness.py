import matplotlib.pyplot as plt
import torch

def hubness_figure(cnt, fig_name):
    cnt = sorted(cnt)
    n = len(cnt)

    x = range(n)
    height = cnt

    plt.bar(x, height)
    plt.title(fig_name)
    plt.savefig(fig_name)
    plt.clf()

class Inverted_softmax():
    # Batchwise inverted softmax.
    # Assume keys do not change over time, so that denominator for each key can be calculated in advance.
    def __init__(self, temperature, probe, keys) -> None:
        self.temperature = temperature
        probe_sim = keys @ probe.T
        self.denominator = torch.sum(torch.exp(probe_sim / self.temperature), dim=1)
        
    def update(self, sims):
        # sims: [B, keys] denominator: [keys]
        return torch.exp(sims / self.temperature) / self.denominator