import logging
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from scipy.special import binom
from joblib import Parallel, delayed
import itertools

logger = logging.getLogger()


class MultipleAgentsComparator:
    """
    Compare sequentially agents, with possible early stopping.
    At maximum, there can be n times K fits done.

    For now, implement only a two-sided test.

    Parameters
    ----------

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

    joblib_backend: str, default = "threading"
        backend to use to parallelize on multi-agents. Use "multiprocessing" or "loky" for a true parallelization.

    Attributes
    ----------
    agent_names: list of str
        list of the agents' names.
    decision: dict
        decision of the tests for each comparison, keys are the comparisons and values are in {"equal", "larger", "smaller"}.

    n_iters: dict
        number of iterations (i.e. number of fits) used for each agent. Keys are the agents' names and values are ints.
    
    Examples
    --------
    One can either use rlberry with self.compare, pre-computed scalars with self.compare_scalar or one can use
    the following code compatible with basically anything:

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

    Where train_evaluate(agent, n) is a function that trains n copies of agent and returns n evaluation values.
    """

    def __init__(
        self,
        n=5,
        K=5,
        B=10000,
        comparisons = None,
        alpha=0.01,
        beta=0,
        n_evaluations=100,
        seed=None,
        joblib_backend="threading",
    ):
        self.n = n
        self.K = K
        self.B = B
        self.alpha = alpha
        self.beta = beta
        self.comparisons = comparisons
        self.n_evaluations = n_evaluations
        self.boundary = []
        self.k = 0
        self.level_spent = 0
        self.power_spent = 0
        self.rng = np.random.RandomState(seed)
        self.rejected_decision = []
        self.joblib_backend = joblib_backend
        self.agent_names = None
        self.current_comparisons = copy(comparisons)
        self.n_iters = None
        self.sum_diffs = None
        self.id_tracked = None

    def compute_sum_diffs(self, k, Z):
        """
        Compute the absolute value of the sum differences.
        """
        comparisons = self.current_comparisons
        boundary = self.boundary
        if k == 0:

            # Define set of permutations. Either all permutations or a random sample if all permutations
            # is too many permutations
            n_permutations = binom(2*self.n, self.n)
            if self.B > n_permutations:
                permutations = itertools.combinations(np.arange(2*self.n), self.n)
                self.normalization = n_permutations
            else:
                permutations = [ np.random.permutation(2*self.n)[:self.n] for _ in range(self.B)]
                self.normalization = self.B

            # Compute the summ differences on the evaluations for each comparisions and for each permutation
            for perm in permutations:
                sum_diff = []
                for i, comp in enumerate(comparisons):
                    Zi = np.hstack([Z[comp[0]][: self.n], Z[comp[1]][: self.n]])
                    mask = np.zeros(2 * self.n)
                    mask[list(perm)] = 1
                    mask = mask == 1
                    sum_diff.append(np.sum(Zi[mask] - Zi[~mask]))
                self.sum_diffs.append(np.array(sum_diff))
        else:
            # Eliminate for conditional
            sum_diffs = []
            for zval in self.sum_diffs:
                if np.max(zval) <= boundary[-1][1]:
                    sum_diffs.append(np.abs(zval))
                    
            # add new permutations. Can be either all the permutations of block k, or using random permutations if this is too many.
            n_permutations = binom(2*self.n, self.n)
            if self.B > n_permutations ** (k+1):
                permutations_k = itertools.combinations(np.arange(2*self.n), self.n)
                self.normalization = n_permutations ** (k+1) # number of permutations
            else :
                n_perm_to_add = len(self.sum_diffs)
                permutations_k = [np.random.permutation(2*self.n)[:self.n] for _ in range(n_perm_to_add)]
                began_random_at = np.floor(np.log(self.B)/np.log(n_permutations))
                self.normalization = n_permutations ** began_random_at # number of permutations

            Zk = np.zeros(2*self.n)
            new_sum_diffs = []
            for id_p, perm_k in enumerate(permutations_k):
                if self.B > n_permutations ** (k+1):
                    perms_before_k = np.arange(n_permutations **k).astype(int)
                else:
                    perms_before_k = [id_p]
                for perm_before_k in perms_before_k:
                    perm_sum_diffs = []
                    for i, comp in enumerate(comparisons):
                        # Compute the sum diffs for given permutation and comparison i
                        Zk[:self.n]=Z[comp[0]][(k * self.n) : ((k + 1) * self.n)]
                        Zk[self.n:(2*self.n)] = Z[comp[1]][(k * self.n) : ((k + 1) * self.n)]
                        mask = np.zeros(2 * self.n)
                        mask[list(perm_k)] = 1
                        mask = mask == 1
                        perm_sum_diffs.append(self.sum_diffs[perm_before_k][i]+ np.sum(Zk[mask] - Zk[~mask]))
                    new_sum_diffs.append(np.array(perm_sum_diffs))
            self.sum_diffs = new_sum_diffs
        return self.sum_diffs

    def partial_compare(self, eval_values, verbose=True):
        """
        Do the test of the k^th interim.

        Parameters
        ----------
        eval_values: dict of agents and evaluations
            keys are agent names and values are concatenation of evaluations till interim k
        verbose: bool
            print Steps
        Returns
        -------
        decision: str in {'accept', 'reject', 'continue'}
           decision of the test at this step.
        T: float
           Test statistic.
        bk: float
           thresholds.
        """
        Z = [eval_values[agent] for agent in eval_values]
        n_managers = len(Z)

        if self.k == 0:
            if self.agent_names is None:
                self.agent_names = list(eval_values.keys())
            # initialization
            if self.comparisons is None:
                self.comparisons = np.array(
                    [(i, j) for i in range(n_managers) for j in range(n_managers) if i < j]
                )
            self.current_comparisons = copy(self.comparisons)
            self.sum_diffs = []
            
            self.n_iters = {self.agent_names[i] : 0 for i in range(n_managers)}
                
            self.decisions = {str(c):"continue" for c in self.comparisons}
            self.id_tracked = np.arange(len(self.decisions)) # ids of comparisons effectively tracked

        k = self.k

        clevel = self.alpha*(k + 1) / self.K
        dlevel = self.beta*(k + 1) / self.K

        rs = np.abs(np.array(self.compute_sum_diffs(k, Z)))

        if verbose:
            print("Step {}".format(k))

        current_decisions = np.array(["continue"]*len(self.id_tracked))
        current_sign = np.zeros(len(current_decisions))
        
        for j in range(len(current_decisions)):
            rs_now = rs[:,current_decisions == "continue"]
            values = np.sort(
                np.max(rs_now, axis=1)
            )  

            icumulative_probas = np.arange(len(rs_now))[::-1] / self.normalization  # This corresponds to 1 - F(t) = P(T > t)

            # Compute admissible values, i.e. values that would not be rejected nor accepted.

            admissible_values_sup = values[
                self.level_spent + icumulative_probas <= clevel
            ]

            if len(admissible_values_sup) > 0:
                bk_sup = admissible_values_sup[0]  # the minimum admissible value
                level_to_add = icumulative_probas[
                    self.level_spent + icumulative_probas <= clevel
                ][0]
            else:
                # This case is possible if clevel-self.level_spent <= 1/ self.normalization (smallest proba possible),
                # in which case there are not enough points and we don't take any decision for now. Happens in particular if B is None.
                bk_sup = np.inf
                level_to_add = 0

            cumulative_probas = np.arange(len(rs_now)) / self.normalization  # corresponds to P(T < t)
            admissible_values_inf = values[
                self.power_spent + cumulative_probas < dlevel
            ]

            if len(admissible_values_inf) > 0:
                bk_inf = admissible_values_inf[-1]  # the maximum admissible value
                power_to_add = cumulative_probas[
                    self.power_spent + cumulative_probas <= dlevel
                ][-1]
            else:
                bk_inf = -np.inf
                power_to_add = 0

            # Test statistic
            Tmax = 0
            Tmin = np.inf
            Tmaxsigned = 0
            Tminsigned = 0
            for i, comp in enumerate(self.current_comparisons[current_decisions == "continue"]):
                Ti = np.abs(
                    np.sum(
                        Z[comp[0]][: ((k + 1) * self.n)]
                        - Z[comp[1]][: ((k + 1) * self.n)]
                    )
                )
                if Ti > Tmax:
                    Tmax = Ti
                    imax = i
                    Tmaxsigned = np.sum(
                        Z[comp[0]][: ((k + 1) * self.n)]
                        - Z[comp[1]][: ((k + 1) * self.n)]
                    )

                if Ti < Tmin:
                    Tmin = Ti
                    imin = i
                    Tminsigned = np.sum(
                        Z[comp[0]][: ((k + 1) * self.n)]
                        - Z[comp[1]][: ((k + 1) * self.n)]
                    )

            if Tmax > bk_sup:
                id_reject = np.arange(len(current_decisions))[current_decisions== "continue"][imax]
                current_decisions[id_reject] = "reject"

                if Tmaxsigned >0:
                    self.decisions[str(self.current_comparisons[id_reject])] = "larger"
                else:
                    self.decisions[str(self.current_comparisons[id_reject])] = "smaller"
                print("reject")
            elif Tmin < bk_inf:
                id_accept = np.arange(len(current_decisions))[current_decisions == "continue"][imin]
                current_decisions[id_accept] = "accept"
                self.decisions[str(self.current_comparisons[id_accept])] = "equal"
            else:
                break

            
        
        self.boundary.append((bk_inf, bk_sup))

        self.level_spent += level_to_add  # level effectively used at this point
        self.power_spent += power_to_add 
        
        if k == self.K - 1:
            for c in self.comparisons:
                if self.decisions[str(c)]=="continue":
                    self.decisions[str(c)] = "equal"
        
        self.k = self.k + 1
        self.eval_values = {self.agent_names[i]: Z[i] for i in range(n_managers)}
        self.mean_eval_values = [np.mean(z) for z in Z]
        for i in range(n_managers):
            self.n_iters[self.agent_names[i]] = len(Z[i].ravel())

        id_decided = np.array(current_decisions) != "continue"

        self.id_tracked = self.id_tracked[~id_decided]
        self.current_comparisons = self.current_comparisons[~id_decided]
        self.sum_diffs = np.array(self.sum_diffs)[:, ~id_decided]

    def compare(self, managers,  clean_after=True, verbose=True):
        """
        Compare the managers for each of the comparisons in `comparisons`.

        Parameters
        ----------
        managers : list of tuple of agent_class and init_kwargs for the agent.
        clean_after: boolean
        verbose: boolean
        """
        
        if not _RLBERRY_INSTALLED :
            raise ValueError("Automatic comparison via `compare` needs the library `rlberry` which is not installed.") 
        
        Z = [np.array([]) for _ in managers]
        # spawn independent seeds, one for each fit and one for the comparator.
        seeder = Seeder(self.rng.randint(10000))
        seeders = seeder.spawn(len(managers) * self.K + 1)
        self.rng = seeders[-1].rng
        
        for k in range(self.K):
            Z = self._fit(managers, Z, k, seeders, clean_after)
            self.partial_compare({self.agent_names[i] : Z[i] for i in range(len(managers))}, verbose)
            decisions = np.array(list(self.decisions.values()))
            if np.all([d in ["smaller", "larger", "equal"] for d in decisions]):
                break

        return self.decisions

    def compare_scalars(self, scalars, agent_names=None):
        """
        Compare the managers for each of the comparisons in `comparisons`.

        Parameters
        ----------
        scalars : list of list of scalars.
        agent_names : list of str or None
        """
        Z = [np.array([]) for _ in scalars]
        if agent_names is None:
            self.agent_names = ["Agent "+str(i) for i in range(len(scalars))]

        for k in range(self.K):
            Z = self._get_z_scalars(scalars, Z, k)
            self.partial_compare({self.agent_names[i] : Z[i] for i in range(len(scalars))}, k)
            decisions = np.array(list(self.decisions.values()))
            if np.all([d in ["smaller", "larger", "equal"] for d in decisions]):
                break
        return self.decisions

    def _get_z_scalars(self, scalars, Z, k):

        for i in range(len(scalars)):
            if (self.current_comparisons is None) or (i in np.array(self.current_comparisons).ravel()):
                Z[i] = np.hstack([Z[i], scalars[i][k * self.n : (k + 1) * self.n]])
        return Z

    def _fit(self, managers, Z, k, seeders, clean_after):
        """
        fit rlberry agents.
        """
        agent_classes = [manager[0] for manager in managers]
        kwargs_list = [manager[1] for manager in managers]
        for kwarg in kwargs_list:
            kwarg["n_fit"] = self.n
        managers_in = []
        for i in range(len(agent_classes)):
            if (self.current_comparisons is None) or (i in np.array(self.current_comparisons).ravel()):
                agent_class = agent_classes[i]
                kwargs = kwargs_list[i]
                seeder = seeders[i]
                managers_in.append(AgentManager(agent_class, **kwargs, seed=seeder))
        if self.agent_names is None:
            self.agent_names = [manager.agent_name for manager in managers_in]

        # For now, paralellize only training because _get_evals not pickleable
        managers_in = Parallel(n_jobs=-1, backend=self.joblib_backend)(
            delayed(_fit_agent)(manager) for manager in managers_in
        )

        idz = 0
        for i in range(len(agent_classes)):
            if (self.current_comparisons is None) or (i in np.array(self.current_comparisons).ravel()):
                Z[i] = np.hstack([Z[i], self._get_evals(managers_in[idz])])
                idz += 1
        if clean_after:
            for m in managers_in:
                m.clear_output_dir()
        return Z

    def _get_evals(self, manager):
        """
        Can be overwritten for alternative evaluation function.
        """
        eval_values = []
        for idx in range(self.n):
            logger.info("Evaluating agent " + str(idx))
            eval_values.append(
                np.mean(manager.eval_agents(self.n_evaluations, agent_id=idx))
            )
        return eval_values

    def plot_results(self, agent_names=None, axes = None):
        """
        visual representation of results.

        Parameters
        ----------
        agent_names : list of str or None
        axes : tuple of two matplotlib axes of None
             if None, use the following:
             `fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(6,5))`
        """

        id_sort = np.argsort(self.mean_eval_values)
        Z = [self.eval_values[self.agent_names[i]] for i  in id_sort]

        if agent_names is None:
            agent_names = self.agent_names

        links = np.zeros([len(agent_names),len(agent_names)])

        for i in range(len(self.comparisons)):
            c = self.comparisons[i]
            decision = self.decisions[str(c)]
            if decision == "equal":
                links[c[0],c[1]] = 0

            elif decision == "larger":
                links[c[0],c[1]] = 1

            else:
                links[c[0],c[1]] = -1


        links = links - links.T
        links = links[id_sort,:][:, id_sort]
        links = links + 2*np.eye(len(links))
        print(links)
        annot = []
        for i in range(len(links)):
            annot_i = []
            for j in range(len(links)):
                if i == j:
                    annot_i.append(" ")                    
                elif links[i,j] == 0:
                    annot_i.append("${\\rightarrow  =}\downarrow$")
                elif links[i,j] == 1:
                    annot_i.append("${\\rightarrow \geq}\downarrow$")
                else:
                    annot_i.append("${\\rightarrow  \leq}\downarrow$")
            annot+= [annot_i]
        if axes is None:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [1, 1]}, figsize=(6,5)
            )
        else:
            (ax1, ax2) = axes

        n_iterations = [self.n_iters[self.agent_names[i]] for i in id_sort]
        the_table = ax1.table(
            cellText=[n_iterations], rowLabels=["n_iter"], loc="top", cellLoc="center"
        )

        # Draw the heatmap with the mask and correct aspect ratio
        res = sns.heatmap(links, annot = annot, cmap="Set2", vmax=2, center=0,linewidths=.5, ax =ax1, 
                          cbar=False, yticklabels=np.array(agent_names)[id_sort],  
                          xticklabels=['']*len(agent_names),fmt='')

        # Drawing the frame
        for _, spine in res.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)

        box_plot = ax2.boxplot(Z, labels=np.array(agent_names)[id_sort], showmeans=True)
        for mean in box_plot['means']:
            mean.set_alpha(0.6)

        ax2.xaxis.set_label([])
        ax2.xaxis.tick_top()

def _fit_agent(manager):
    manager.fit()
    return manager
