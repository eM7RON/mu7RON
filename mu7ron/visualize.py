from matplotlib import pyplot as plt
from matplotlib import style


def bar_chart(
    adict,
    x_label="",
    y_label="",
    fig_title="",
    log_y=False,
    sort_y=False,
    reverse_sort=False,
    rotate_y=False,
    tck_fmt="sci",
    **kwargs
):

    style.use("seaborn")

    if sort_y:
        alist = sorted(adict.items(), key=lambda x: x[1], reverse=reverse_sort)
    else:
        alist = sorted(adict.items(), key=lambda x: x[0], reverse=reverse_sort)
    labels = []
    counts = []
    for tup in alist:
        labels.append(tup[0])
        counts.append(tup[1])
    x_pos = list(range(len(labels)))

    plt.figure(figsize=kwargs.get("figsize", (25, 6)))
    plt.bar(
        x_pos,
        counts,
        alpha=0.7,
        color=[
            "#0b1923",
            "#ff1188",
            "#ffee22",
            "#11bbcc",
            "#004150",
            "#087436",
            "#ffc300",
            "#ec5b1d",
            "#c70039",
            "#221131",
            "#12a3a7",
            "#a6159f",
        ],
        ec="w",
        # log=True,
    )
    plt.xticks(x_pos, labels, rotation=45, fontsize=10)
    plt.yticks(fontsize=12)
    plt.gca().ticklabel_format(axis="y", style=tck_fmt, scilimits=(0, 0))
    if log_y:
        plt.gca().set_yscale("log")
        plt.xlabel(x_label, fontsize=16)
    if rotate_y:
        plt.ylabel(y_label, fontsize=16, rotation=0)
        plt.gca().yaxis.set_label_coords(-0.035, 0.5)
    else:
        plt.ylabel(y_label, fontsize=16)
    plt.title(fig_title, fontsize=18)
    plt.show()
