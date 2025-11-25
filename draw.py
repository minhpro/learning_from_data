import matplotlib.pyplot as plt
import numpy as np

def draw_line(x, line, color, prefix):
    # Generate x values
    a = line[1]
    b = line[2]
    c = line[0]

    # Compute y values using y = (-a*x - c)/b, if b != 0
    if b != 0:
        y = (-a * x - c) / b
        plt.plot(x, y, label=f'{prefix}: {a}x + {b}y + {c} = 0')
    else:
        # Vertical line case: x = -c/a
        x_line = -c / a
        plt.axvline(x=x_line, color=color, label=f'x = {x_line}')


def draw_line_and_samples(f, g, samples, test_samples, size, test_size):
    # Generate x values
    axis = np.linspace(-100, 100, 400)
    draw_line(axis, f, 'r', 'f')
    draw_line(axis, g, 'm', 'g')
    xs, ys = samples
    xt, yt = test_samples 

    for i in range(size):
        if ys[i] == 1:
            plt.scatter(xs[i, 1], xs[i, 2], color='red', zorder=5)
        else:
            plt.scatter(xs[i, 1], xs[i, 2], color='blue', zorder=5)
    
    for i in range(test_size):
        if yt[i] == 1:
            plt.scatter(xt[i, 1], xt[i, 2], color='orange', zorder=5)
        else:
            plt.scatter(xt[i, 1], xt[i, 2], color='green', zorder=5)
    # Customize plot
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Line: ax + by + c = 0')
    plt.grid(True)
    plt.show()