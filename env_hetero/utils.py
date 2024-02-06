import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time

def plotQseq(Qtk_list, Qstar, Qkstar):
    # creating initial data values
    # of x and y
    mpl.use('macosx')
    x = range(Qstar.shape[0])
    y = Qtk_list[0]

    # to run GUI event loop
    plt.ion()

    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x, y, 'o')

    # setting title
    plt.title("Convergence of Q learning", fontsize=20)

    # setting x-axis label and y-axis label
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.plot(Qkstar, 'o', color='red', alpha=0.8, label=f"Q0 star")
    plt.plot(Qstar, 'o', color='green', alpha=0.2, label=f"Q star")
    L = plt.legend()
    # Loop
    for i, Qtk in enumerate(Qtk_list):
        # creating new Y values
        new_y = Qtk

        # updating data values
        line1.set_xdata(x)
        line1.set_ydata(new_y)
        L.get_texts()[0].set_text(f"Qtk, t = {i}")
        line1.set_alpha(0.5)

        # drawing updated values
        figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()

        time.sleep(0.1)


if __name__ == "__main__":
    mpl.use('macosx')





