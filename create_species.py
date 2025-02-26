import numpy as np
import pandas as pd
import hashlib


class SpeciesClass:
    def __init__(self, A, B, traitsA, traitsB, Aabund, Babund, ADV, BDV, spec):
        self.A = A
        self.B = B
        self.traitsA = traitsA
        self.traitsB = traitsB
        self.Aabund = Aabund
        self.Babund = Babund
        self.ADV = ADV
        self.BDV = BDV
        self.spec = spec


def create_discrete(rangeDiscrete, rng):
    """
    Gera uma função de amostragem discreta baseada no intervalo fornecido.
    """
    Nlevels = rng.choice(rangeDiscrete, 1)[0]
    prob = rng.integers(1, Nlevels + 2, size=Nlevels)
    prob_normalized = prob / prob.sum()

    def discrete_sampling(n):
        return rng.choice(range(1, Nlevels + 1), size=n, p=prob_normalized, replace=True)

    return discrete_sampling


def create_species(NumberA, NumberB, traitsA, traitsB, abundance, specRange,
                   speciesClass=None, specialist=True, coLin=None):

    seed_input = f"{NumberA}_{NumberB}_{traitsA}_{traitsB}_{specRange}_{specialist}"
    derived_seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(derived_seed)

    spec = specialist


    if speciesClass is not None:
        traitsA = speciesClass.traitsA
        traitsB = speciesClass.traitsB
        ADV = speciesClass.ADV
        BDV = speciesClass.BDV
    else:
        ADV = []
        BDV = []


    A = pd.DataFrame(np.nan, index=range(NumberA), columns=range(sum(traitsA)))
    B = pd.DataFrame(np.nan, index=range(NumberB), columns=range(sum(traitsB)))

    A.index = [f'a{x}' for x in range(1, NumberA + 1)]
    B.index = [f'b{x}' for x in range(1, NumberB + 1)]
    A.columns = [f'A{x}' for x in range(1, sum(traitsA) + 1)]
    B.columns = [f'B{x}' for x in range(1, sum(traitsB) + 1)]

    sampling = lambda n: rng.uniform(0, 1, n)


    if traitsA[1] != 0:
        A.iloc[:, :traitsA[1] + traitsA[0]] = np.array([sampling(NumberA) for _ in range(traitsA[1])]).T
    if traitsB[1] != 0:
        B.iloc[:, :traitsB[1] + traitsB[0]] = np.array([sampling(NumberB) for _ in range(traitsB[1])]).T


    if speciesClass is None:
        for i in range(traitsA[0]):
            ADV.append(create_discrete(range(2, 9), rng))
            A.iloc[:, i] = ADV[-1](NumberA)

        for i in range(traitsB[0]):
            BDV.append(create_discrete(range(2, 9), rng))
            B.iloc[:, i] = BDV[-1](NumberB)


    if callable(abundance):
        Aabund = abundance(NumberA, NumberB, rng)
        Babund = abundance(NumberB, NumberA, rng)
    else:
        Aabund = rng.poisson(abundance, NumberA) + 1
        Babund = rng.poisson(abundance, NumberB) + 1


    if isinstance(spec, bool) and spec:
        spec = rng.uniform(specRange[0], specRange[1], NumberB)
    elif isinstance(spec, bool) and not spec:
        spec = np.ones(NumberB)

    return SpeciesClass(A, B, traitsA, traitsB, Aabund, Babund, ADV, BDV, spec)
