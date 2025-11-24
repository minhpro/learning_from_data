from draw import draw_line, draw_line_and_samples
from sample import generate_target_function, generate_samples
from pla import pla, pocket
import numpy as np

INPUT_D = 2
SAMPLE_SIZE = 20
TEST_SIZE = 20

f = generate_target_function(INPUT_D)
print(f"{f[1]}x + {f[2]}y + {f[0]} = 0")
samples = generate_samples(INPUT_D, f, SAMPLE_SIZE)
test_samples = generate_samples(INPUT_D, f, TEST_SIZE)
g = pocket(samples, SAMPLE_SIZE, np.zeros(INPUT_D + 1))
print(f"{g[1]}x + {g[2]}y + {g[0]} = 0")

draw_line_and_samples(f, g, samples, test_samples, SAMPLE_SIZE, TEST_SIZE)

