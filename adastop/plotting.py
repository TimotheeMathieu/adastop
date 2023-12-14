import matplotlib.pyplot as plt


def plot_results(comparator, agent_names=None, axes = None):
    """
    visual representation of results.

    Parameters
    ----------
    agent_names : list of str or None
    axes : tuple of two matplotlib axes of None
         if None, use the following:
         `fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(6,5))`
    """

    id_sort = np.argsort(comparator.mean_eval_values)
    Z = [comparator.eval_values[comparator.agent_names[i]] for i  in id_sort]

    if agent_names is None:
        agent_names = comparator.agent_names

    links = 2*np.ones([len(agent_names),len(agent_names)]) # all initialized with no decision

    for i in range(len(comparator.comparisons)):
        c = comparator.comparisons[i]
        decision = comparator.decisions[str(c)]
        if decision == "equal":
            links[c[0],c[1]] = 0
            links[c[1],c[0]] = 0
        elif decision == "larger":
            links[c[0],c[1]] = 1
            links[c[1],c[0]] = -1
        else:
            links[c[0],c[1]] = -1
            links[c[1],c[0]] = 1

    links = links[id_sort,:][:, id_sort]

    annot = []
    for i in range(len(links)):
        annot_i = []
        for j in range(len(links)):
            if links[i,j] == 2:
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

    n_iterations = [comparator.n_iters[comparator.agent_names[i]] for i in id_sort]
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

def plot_results_sota(comparator, agent_names=None, axes = None):
    """
    visual representation of results -- all versus one. The one is supposed to be first agent name.

    Parameters
    ----------
    agent_names : list of str or None
    axes : tuple of two matplotlib axes of None
         if None, use the following:
         `fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(6,5))`
    """
    id_sort = np.argsort(comparator.mean_eval_values)
    Z = [comparator.eval_values[comparator.agent_names[i]] for i  in id_sort]

    if agent_names is None:
        agent_names = comparator.agent_names
    assert len(comparator.decisions) == len(agent_names) -1

    links = 2*np.ones([len(agent_names)]) # all initialized with no decision


    for i in range(len(comparator.comparisons)):
        c = comparator.comparisons[i]
        decision = comparator.decisions[str(c)]
        if decision == "equal":
            links[c[1]] = 0
        elif decision == "larger":
            links[c[1]] = 1
        else:
            links[c[1]] = -1

    links = links[id_sort]

    annot = []
    for i in range(len(links)):
        if links[i] == 2:
            annot.append(" ")                    
        elif links[i] == 0:
            annot.append("$=$")
        elif links[i] == 1:
            annot.append("$\geq$")
        else:
            annot.append("$\leq$")
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 6]}, figsize=(6,3.5)
        )
    else:
        (ax1, ax2) = axes

    n_iterations = [comparator.n_iters[comparator.agent_names[i]] for i in id_sort]
    the_table = ax1.table(
        cellText=[n_iterations], rowLabels=["$N_{scores}$"], loc="top", cellLoc="center",
    )
    for c in the_table.get_celld().values():
        c.visible_edges = ''

    print(links, annot)
    # Draw the heatmap with the mask and correct aspect ratio
    res = sns.heatmap([np.array(links)[:-1]], annot = [np.array(annot)[:-1]], cmap="Set2", vmax=2, center=0,linewidths=.5, ax =ax1, cbar=False,yticklabels=[''], xticklabels=['']*len(agent_names),fmt='')

    def set_xmargin(ax, left=0.0, right=0.3):
        ax.set_xmargin(0)
        ax.autoscale_view()
        lim = ax.get_xlim()
        delta = np.diff(lim)
        left = lim[0] - delta*left
        right = lim[1] + delta*right
        ax.set_xlim(left,right)

    set_xmargin(ax1, right=0.14)

    # Drawing the frame
    for _, spine in res.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    box_plot = ax2.boxplot(Z, labels=np.array(agent_names)[id_sort], showmeans=True)
    for mean in box_plot['means']:
        mean.set_alpha(0.6)

    ax2.xaxis.set_label([])
    ax2.xaxis.tick_top()
    plt.subplots_adjust(top=0.9, hspace=0.3)

