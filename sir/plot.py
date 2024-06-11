import matplotlib.pyplot as plt

def plot_sir(solution, t, data):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, solution[:, 0], 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, solution[:, 1], 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, solution[:, 2], 'g', alpha=0.5, lw=2, label='Recovered')
    ax.scatter(t, data, color='k', label='Original data')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.set_ylim(0,2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
