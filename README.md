# adastop
Sequential testing for efficient andreliable comparison of stochastic algorithms.
# compare_agents module


### _class_ compare_agents.MultipleAgentsComparator(n=5, K=5, B=10000, comparisons=None, alpha=0.01, beta=0, n_evaluations=100, seed=None, joblib_backend='threading')
Bases: `object`

Compare sequentially agents, with possible early stopping.
At maximum, there can be n times K fits done.

For now, implement only a two-sided test.

n: int, default=5

    number of fits before each early stopping check

K: int, default=5

    number of check

B: int, default=None

    Number of random permutations used to approximate permutation distribution.

comparisons: list of tuple of indices or None

    if None, all the pairwise comparison are done.
    If = [(0,1), (0,2)] for instance, the compare only 0 vs 1  and 0 vs 2

alpha: float, default=0.01

    level of the test

beta: float, default=0

    power spent in early accept.

n_evaluations: int, default=100

    number of evaluations used in the function _get_rewards.

seed: int or None, default = None

joblib_backend: str, default = “threading”

    backend to use to parallelize on multi-agents. Use “multiprocessing” or “loky” for a true parallelization.

agent_names: list of str

    list of the agents’ names.

decision: dict

    decision of the tests for each comparison, keys are the comparisons and values are in {“equal”, “larger”, “smaller”}.

n_iters: dict

    number of iterations (i.e. number of fits) used for each agent. Keys are the agents’ names and values are ints.

One can either use rlberry with self.compare, pre-computed scalars with self.compare_scalar or one can use
the following code compatible with basically anything:

```python
>>> comparator = Comparator(n=6, K=6, B=10000, alpha=0.05, beta=0.01)
>>>
>>> eval_values = {agent.name: [] for agent in agents}
>>>
>>> for k in range(comparator.K):
>>>    for  agent in enumerate(agents):
>>>        # If the agent is still in one of the comparison considered, then generate new evaluations.
>>>        if agent in comparator.current_comparisons.ravel():
>>>            eval_values[agent.name].append(train_evaluate(agent, n))
>>>    decisions, T = comparator.partial_compare(eval_values, verbose)
>>>    if np.all([d in ["accept", "reject"] for d in decisions]):
>>>        break
```

Where train_evaluate(agent, n) is a function that trains n copies of agent and returns n evaluation values.


#### \__init__(n=5, K=5, B=10000, comparisons=None, alpha=0.01, beta=0, n_evaluations=100, seed=None, joblib_backend='threading')

#### compare(managers, clean_after=True, verbose=True)
Compare the managers for each of the comparisons in comparisons.

managers : list of tuple of agent_class and init_kwargs for the agent.
clean_after: boolean
verbose: boolean


#### compare_scalars(scalars, agent_names=None)
Compare the managers for each of the comparisons in comparisons.

scalars : list of list of scalars.
agent_names : list of str or None


#### compute_sum_diffs(k, Z)
Compute the absolute value of the sum differences.


#### partial_compare(eval_values, verbose=True)
Do the test of the k^th interim.

eval_values: dict of agents and evaluations

    keys are agent names and values are concatenation of evaluations till interim k

verbose: bool

    print Steps

decision: str in {‘accept’, ‘reject’, ‘continue’}

    decision of the test at this step.

T: float

    Test statistic.

bk: float

    thresholds.


#### plot_results(agent_names=None, axes=None)
visual representation of results.

agent_names : list of str or None
axes : tuple of two matplotlib axes of None

> if None, use the following:
> fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={“height_ratios”: [1, 2]}, figsize=(6,5))
