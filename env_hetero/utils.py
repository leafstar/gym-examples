import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time


def plotQseq(Qtk_list, Qstar, Qkstar=None, Deltas=None):
    # creating initial data values
    # of x and y
    mpl.use('macosx')
    x = range(Qstar.shape[0])
    y = Qtk_list[0]

    # to run GUI event loop
    plt.ion()

    # here we are creating subplots
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x, y, 'o', label=r"$\barQ_t^k$")
    ax.ylim = np.max(Qtk_list)
    # setting title
    plt.title("Convergence of Q learning", fontsize=20)

    # setting x-axis label and y-axis label
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    if Qkstar is not None:
        plt.plot(Qkstar, 'o', color='red', alpha=0.8, label=f"Q0 star")
    plt.plot(Qstar, 'o', color='indigo', alpha=0.8, label=f"Q star")
    L = plt.legend()
    # Loop
    for i, Qtk in enumerate(Qtk_list):
        # creating new Y values
        new_y = Qtk

        # updating data values
        line1.set_xdata(x)
        line1.set_ydata(new_y)
        if Deltas is not None:
            L.get_texts()[0].set_text(r"$\bar Q_t^k$," + f" t = {i}, Delta = {Deltas[i]}")
        else:
            L.get_texts()[0].set_text(r"$\bar Q_t^k$," + f" t = {i}")
        line1.set_alpha(0.5)

        # drawing updated values
        figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()
        time.sleep(0.000002)
        # if i > 1000:
        #     time.sleep(0.00001)
        # else:
        #     time.sleep(0.002)


def analytical(t):
    """
    Analytical solution for exponential decay for this recurrence relation.
    Delta_{t+1} <= (1-(1-gamma)*lam_t)*Delta_{t}
    Args:
        t:
        delta0: Initial difference
        gamma: discount factor
    Returns:
        a list: [Delta_{0}, Delta_{1},..., Delta_{t}]
    """

    gamma = 0.99
    a = 1
    lam = np.log(t) ** a / t ** 0.8
    E = np.log(t)
    kappa = 1.2
    res1 = 32 / 3 / (1 - gamma) ** 2 * np.exp(-0.5 * np.sqrt((1 - gamma) * t * lam))
    res2 = 32 / 3 / (1 - gamma) ** 2 * gamma ** 2 * kappa * lam * E
    res3 = 16 * E / (1 - gamma) ** 3 / (t + E)
    # print(t,np.prod(1-np.array([lam(i) for i in range(t)])*(1-gamma)),delta0)
    # return np.prod(1 - np.array([lam(i) for i in range(t)]) * (1 - gamma)) * delta0
    return res1, res2, res3


def lambda_analysis(lambdas, mid):
    res = 0
    t = len(lambdas)
    # for i in range(mid + 1):
    #     prod = 1
    #     for j in range(i + 1, t):
    #         prod *= 1 - lambdas[j]
    #     res += lambdas[i] * prod
    res2 = 0
    for i in range(mid + 1, t):
        prod = 1
        for j in range(i + 1, t):
            prod *= 1 - lambdas[j]
        res2 += lambdas[i] * prod
    return res, res2


if __name__ == "__main__":
    mpl.use('macosx')
    print(analytical(1e9))
