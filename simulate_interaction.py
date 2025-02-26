from create_species import *
from scipy.stats import norm

def check_weights(weights, main_len, inter_len):
    """Verifica se os pesos fornecidos são válidos"""
    if "main" not in weights or "inter" not in weights:
        raise ValueError("Weights dictionary must contain keys 'main' and 'inter'")

    if not isinstance(weights["main"], (list, tuple)) or not isinstance(weights["inter"], (list, tuple)):
        raise ValueError("Weights must be provided as lists or tuples")

    if len(weights["main"]) != main_len:
        raise ValueError("Length of 'weights[\"main\"]' must match the length of 'main'")

    if len(weights["inter"]) != inter_len:
        raise ValueError("Length of 'weights[\"inter\"]' must match the number of interactions in 'inter'")

def check_species(species, kwargs):
    """Verifica a existência e tipo de 'species'"""
    if species is None:
        species = create_species(**kwargs)
    if not isinstance(species, SpeciesClass):
        print("species must be of class SpeciesClass")
    return species

def calculate_trait_value(val, x, discrete, mainTraits, spec=0.5):
    """Calcula o valor de um traço com base se é discreto ou contínuo"""
    if mainTraits[val]["discrete"]:
        return mainTraits[val]["mean"][x[val]] * mainTraits[val]["weight"]
    else:
        return np.random.normal(mainTraits[val]["mean"], spec) * mainTraits[val]["weight"]

def create_distribution_func(interMatrix, is_poisson=True, x=1000, seed=42):
    """Cria uma função genérica para distribuir valores de Poisson ou binários"""
    np.random.seed(seed)
    data = np.random.poisson(interMatrix * x).astype(float) if is_poisson else (np.random.poisson(interMatrix * x) > 0).astype(int)
    return pd.DataFrame(data, columns=range(data.shape[1]), index=range(data.shape[0]))

def calculate_interactions_logic(i, x, y, inter, interTraits, spec=0.5):
    """Calcula o valor da interação para um par de espécies baseado nos traços"""
    if interTraits[i]['both']:
        return interTraits[i]['interM'][x[0, np.where(inter == inter[i, 0])[0][0]], y[0, np.where(inter == inter[i, 1])[0][0]]] * interTraits[i]['weight']
    elif interTraits[i]['which'] != 3:
        if interTraits[i]['which'] == 1:
            return norm.pdf(y[0, np.where(inter == inter[i, 1])[0][0]], loc=interTraits[i]['mean'][x[0, np.where(inter == inter[i, 0])[0][0]]], scale=spec) * interTraits[i]['weight']
        elif interTraits[i]['which'] == 2:
            return norm.pdf(x[0, np.where(inter == inter[i, 0])[0][0]], loc=interTraits[i]['mean'][y[0, np.where(inter == inter[i, 1])[0][0]]], scale=spec) * interTraits[i]['weight']
    else:
        return norm.pdf(np.log(x[0, np.where(inter == inter[i, 0])[0][0]] / y[0, np.where(inter == inter[i, 1])[0][0]]), loc=0, scale=spec) * interTraits[i]['weight']

def calculate_interactions(species, mainFunc, interFunc, inter, interTraits, weights, interMatrix):
    """Calcula a matriz de interações entre as espécies, com otimização de operações de ponto flutuante"""


    Aabund = species.Aabund
    Babund = species.Babund

    random_values = np.random.uniform(0.3, 0.7, size=(species.A.shape[0], species.B.shape[0]))

    for i in range(species.A.shape[0]):
        x = species.A.iloc[i, :].to_numpy().reshape(1, -1)
        for j in range(species.B.shape[0]):
            y = species.B.iloc[j, :].to_numpy().reshape(1, -1)
            spec = species.spec[j]

            interMatrix[i, j] = (
                mainFunc(x, y, spec) *
                interFunc(x, y, inter, interTraits, spec) *
                Aabund[i] *
                Babund[j] *
                random_values[i, j]
            )

    return interMatrix

def simulate_interaction(species, main, inter, weights={ "main": [],"inter": []}, re_sim= None, **kwargs):

    check_weights(weights, len(main), inter.shape[0])
    species = check_species(species, kwargs)

    discrete = []
    if species.traitsA != 0 and species.traitsB != 0:
        discrete = species.A.columns[:species.traitsA[0]].tolist() + species.B.columns[:species.traitsB[0]].tolist()

    def createCov():
        return np.array([[1, np.random.uniform(-0.5, 0.5)], [np.random.uniform(-0.5, 0.5), 1]])

    mainTraits = {
        m: {'discrete': m in discrete, 'mean': np.random.uniform(0, 1, 20).tolist() if m in discrete else 0, 'weight': w}
        for m, w in zip(main, weights["main"])
    }

    def mainFunc(x, y, spec=0.5):
        return np.prod([calculate_trait_value(val, x, discrete, mainTraits, spec) for val in main])


    interTraits = []
    for i, paar in enumerate(inter):
        whichD = [elem in discrete for elem in paar]
        if sum(whichD) > 1:
            interTraits.append({'both': True, 'interM': np.random.uniform(0, 1, size=(10, 10)), 'weight': weights['inter'][i]})
        else:
            interTraits.append({'both': False, 'which': (1 if any(whichD) else 3), 'mean': np.random.normal(0, 1, size=10), 'weight': weights['inter'][i]})


    def interFunc(x, y, inter, interTraits, spec=0.5):
        if inter is not None:
            res = np.array([calculate_interactions_logic(i, x, y, inter, interTraits, spec) for i in range(len(interTraits))])
            return np.prod(res)
        else:
            return 1

    interMatrix = np.full((species.A.shape[0], species.B.shape[0]), np.nan)
    interMatrix = calculate_interactions(species, mainFunc, interFunc, inter, interTraits, weights, interMatrix)


    if species.traitsA[0] != 0:
        species.A[species.A.columns[:species.traitsA[0]]] = species.A[species.A.columns[:species.traitsA[0]]].astype('category')

    if species.traitsB[0] != 0:
        species.B[species.B.columns[:species.traitsB[0]]] = species.B[species.B.columns[:species.traitsB[0]]].astype('category')

    out = {
        'A': species.A.reset_index().rename(columns={'index': 'X'}),
        'B': species.B.reset_index().rename(columns={'index': 'Y'}),
        'mainFunc': mainFunc,
        'interFunc': interFunc,
        'poisson': lambda x: create_distribution_func(interMatrix, is_poisson=True, x=x),
        'binar': lambda x: create_distribution_func(interMatrix, is_poisson=False, x=x),
        'species': species,
        'z': pd.DataFrame(interMatrix, index=species.A.index, columns=species.B.index),
        'settings': {'interT': interTraits, 'inter': inter, 'mainT': mainTraits, 'main': main, 'interMatrix': interMatrix}
    }

    if re_sim is not None:
        species = create_species(speciesClass=re_sim['species'], **kwargs)
        interMatrix = calculate_interactions(species, re_sim['mainFunc'], re_sim['interFunc'], inter, weights, interMatrix)
        out['mainFunc'] = re_sim['mainFunc']
        out['interFunc'] = re_sim['interFunc']

    return out

def minOneInter(inter):
    inter = np.array(inter)
    which_col = np.any(inter == 1, axis=0)
    which_row = np.any(inter == 1, axis=1)
    result = inter[which_row][:, which_col]
    return pd.DataFrame(result)