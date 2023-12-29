import torch
import torch.nn.functional as F
from itertools import product
import warnings
import math
from tqdm import tqdm, trange
import random
import functools
from collections import namedtuple

need = {(0.0, 0.0): 1, (0.0, 1.0): 1, (1.0, 0.0): 1, (1.0, 1.0): 0}
num_vars = 2

operation = namedtuple('operation', ['repr', 'op'])
ops = (
    operation("+", lambda x, y: x + y),
    operation("-", lambda x, y: x - y),
    operation("×", lambda x, y: x * y),
    operation("max", torch.maximum),
    operation("min", torch.minimum),
)

acts = (
    F.relu,
    F.hardtanh,
    F.hardswish,
    F.relu6,
    F.elu,
    F.selu,
    F.celu,
    F.leaky_relu,
    F.rrelu,
    F.gelu,
    F.logsigmoid,
    F.hardshrink,
    F.tanhshrink,
    F.softsign,
    F.softplus,
    F.softshrink,
    F.tanh,
    F.sigmoid,
    F.hardsigmoid,
    F.silu,
    F.mish,
    torch.neg,
    # torch.sign,
    torch.sin,
    torch.cos,
    torch.sinh,
    torch.sqrt,
    torch.square,
    torch.tan,
    # torch.trunc
)


def stochastic(n: int,
               k: int,
               δ: float = 0.1,
               α: torch.Tensor = torch.tensor([0., 1., 0., 1.]),
               β: torch.Tensor = torch.tensor([0., 0., 1., 1.]),
               target: torch.Tensor = torch.tensor([1., 1., 1., 0.]),
               output: bool = False,
               quiet: bool = False):
    """
    :param n: Number of functions to try
    :param k: Number of operations in each function
    :param δ: Maximum acceptable error
    :param α: Values of first argument
    :param β: Values of second argument
    :param target: Desired values of function on α and β
    :param output: Whether to return found functions
    :param quiet: If True then use range; else use trange
    :return:
    """
    # number of functions which match the target exactly
    exact = 0
    # number of functinos which are within error δ of the target
    near = 0
    # namedtuple for the return type to contain all the information
    logicfunction = namedtuple('LogicFunction', ['error', 'function', 'repr'])
    found = []

    for _ in (range(n) if quiet else trange(n)):
        # the aggregation operator to use
        op = random.choice(ops)
        # list of actions to take
        actions = random.choices(acts, k=k)
        # which actions occur before or after the aggregation
        lo, hi = sorted((random.randint(0, k), random.randint(0, k)))
        # the resulting function
        fun = lambda α, β: functools.reduce(
            lambda x, f: f(x), (actions[i] for i in range(hi, k)), op.op
            (functools.reduce(lambda x, f: f(x),
                              (actions[i] for i in range(lo)), α),
             functools.reduce(lambda x, f: f(x),
                              (actions[i] for i in range(lo, hi)), β)))
        # error compared to the target
        ϵ = torch.sum(torch.abs(target - fun(α, β))).item()
        if ϵ < δ:
            near += 1
            if output:
                α2 = functools.reduce(lambda x, f: f"{f.__name__}({x})",
                                      (actions[i] for i in range(lo)), "α")
                β2 = functools.reduce(lambda x, f: f"{f.__name__}({x})",
                                      (actions[i] for i in range(lo, hi)), "β")
                # string representation of the function
                eq = functools.reduce(lambda x, f: f"{f.__name__}({x})",
                                      (actions[i] for i in range(hi, k)),
                                      f"{α2} {op.repr} {β2}")
                found.append(logicfunction(ϵ, fun, eq))
            if ϵ == 0:
                exact += 1
    return (exact, near, found) if output else (exact, near)


def enumerate_operations(
               k: int,
               δ: float = 0.1,
               α: torch.Tensor = torch.tensor([0., 1., 0., 1.]),
               β: torch.Tensor = torch.tensor([0., 0., 1., 1.]),
               target: torch.Tensor = torch.tensor([1., 1., 1., 0.]),
               output: bool = False,
               quiet: bool = False):
    """
    :param k: Number of operations in each function
    :param δ: Maximum acceptable error
    :param α: Values of first argument
    :param β: Values of second argument
    :param target: Desired values of function on α and β
    :param output: Whether to return found functions
    :param quiet: If True then use range; else use trange
    :return:
    """

    results = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a1, a2, a3 in tqdm(product(acts, repeat=3), total=len(acts)**3):
            for cname, cop in ops:
                try:
                    ϵ = torch.sum(torch.abs(a3(cop(a1(α), a2(β))) - target)).item()
                    if math.isfinite(ϵ) and ϵ < 0.1:
                        results.append(
                            f"{ϵ=} {a3.__name__}({a1.__name__}(α) {cname} {a2.__name__}(β))"
                        )
                except:
                    continue

    reses = sorted(results)

    for result in reses:
        print(result)

    print(len(reses), "solutions")


def stochastic_symmetric(n: int, k: int, δ: float = 0.1, output: bool = False):
    """
    :param n: the number of possible functions to check
    :param k: the number of components in each function
    :param δ: the maximum acceptable error
    :param output: whether to return results once found
    :return:
    """
    α = torch.tensor([0., 1., 0., 1.])
    β = torch.tensor([0., 0., 1., 1.])
    γ = torch.tensor([1., 1., 1., 0.])
    exact = 0
    near = 0
    for _ in trange(n):
        op = random.choice(ops)
        mid = random.randint(0, k // 2)
        actions = random.choices(acts, k=k - mid)
        α2 = functools.reduce(lambda x, f: f(x),
                              (actions[i] for i in range(mid)), α)
        β2 = functools.reduce(lambda x, f: f(x),
                              (actions[i] for i in range(mid)), β)
        γ2 = functools.reduce(lambda x, f: f(x),
                              (actions[i] for i in range(mid, k - mid)),
                              op.op(α2, β2))
        ϵ = torch.sum(torch.abs(γ - γ2)).item()
        if ϵ < δ:
            near += 1
            if output:
                ops1 = functools.reduce(lambda x, f: f"{f.__name__} ∘ {x}",
                                        (actions[i] for i in range(mid)), "id")
                γ2 = functools.reduce(lambda x, f: f"{f.__name__} ∘ {x}",
                                      (actions[i]
                                       for i in range(mid, k - mid)),
                                      f"{op.repr} ∘ ⟨{ops1}, ...⟩(α, β)")
                tqdm.write(f"{ϵ=} {γ2}")
            if ϵ == 0:
                exact += 1
    return exact, near


def plot_all(n: int, δ: float):
    import numpy as np
    import matplotlib.pyplot as plt

    xs = np.arange(2, 11, dtype=int)
    ys1 = []
    ys2 = []
    outfile = open('./logic-nums.txt', 'a')
    for k in xs:
        exact, near = stochastic(n, k, δ)
        ys1.append(exact * len(acts)**k * len(ops) / n)
        ys2.append(near * len(acts)**k * len(ops) / n)
        outfile.write(f'{n=} {k=} {δ=} {exact=} {near=}\n')
    plt.rcParams.update({'font.size': 16})
    plt.title('NAND gates found')
    plt.xlabel('Number of operations')
    plt.ylabel('Number of NAND gates')
    plt.yscale('log')
    plt.plot(xs, ys2, label='$\\epsilon<0.1$')
    plt.plot(xs, ys1, label='exact')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
