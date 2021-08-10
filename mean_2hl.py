from functools import reduce
from operator import mul

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out as out
from scipy.stats import linregress
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

wd = WorkingDirectory("_experiments", "2hl")


def forward_2hl(w3, w2, b2, w1, b1, x):
    width = B.shape(w2, -1)
    phi = B.tanh
    z = phi(B.matmul(w1, x) + b1)
    z = phi(B.matmul(w2, z) / B.sqrt(width) + b2)
    return B.matmul(w3, z) / B.sqrt(width)


def forward_2hl_sample(batch, width, x):
    x = B.flatten(B.uprank(x, rank=1))

    w3 = B.randn(batch, 1, width)
    w2 = B.randn(batch, width, width)
    w1 = B.randn(batch, width, B.length(x))
    b2 = B.randn(batch, width, 1)
    b1 = B.randn(batch, 1, 1)

    def numel(x):
        return reduce(mul, B.shape(x)[1:], 1)

    n = numel(w3) + numel(w2) + numel(w1) + numel(b2) + numel(b1)

    def bias(x):
        return x + 10 / B.sqrt(n)

    w3 = bias(w3)
    w2 = bias(w2)
    w1 = bias(w1)
    b2 = bias(b2)
    b1 = bias(b1)

    return B.flatten(forward_2hl(w3, w2, b2, w1, b1, x[None, :, None]))


x = 0.5
out.kv("Fixed input", x)

widths = []
values = []

batch_size = 1
batches = 5000
iters = 10

out.kv("Batch size", batch_size)
out.kv("Batches", batches)

for width in np.logspace(np.log10(10), np.log10(1000), iters):
    with out.Section("Iteration"):
        width = int(width)
        out.kv("Width", width)
        mean = np.mean(
            [B.mean(forward_2hl_sample(batch_size, width, x)) for _ in range(batches)]
        )
        out.kv("Value", mean)

    widths.append(width)
    values.append(B.abs(mean))


plt.figure()
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.plot(widths, values, marker="o")
slope, intercept, _, _, _ = linregress(np.log(widths), np.log(values))
x = np.linspace(10, 1000, 200)
plt.plot(
    x,
    np.exp(slope * np.log(x) + intercept),
    label=f"$y \propto x^{{{slope:.2f}}}$",
)
plt.xlabel("Width")
plt.ylabel("Absolute value of mean at some input")
plt.title("Two Hidden Layers")
tweak()
plt.savefig(wd.file("graph.pdf"))
plt.show()
