import pathlib
from typing import Sequence

import numpy as np
from matplotlib import figure
from matplotlib import pyplot as plt

from phase3 import collection
from phase3 import target


def create_comms_plot(
    transmissions: Sequence[collection.Transmission],
    targets: list[target.Target],
    sat_names: list[str],
    super_title: str,
    y_label: str,
    filename: str,
    prefix: str | None = None,
    save: bool = True,
) -> figure.Figure:
    # Create a figure
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(super_title, fontsize=14)
    ax = fig.add_subplot(111)

    prev_data, target_sent_data, target_rec_data = collection.aggregate_transmissions(
        transmissions,
        sat_names,
    )

    count = 0
    for target in targets:
        target_id = target.targetID
        sent_data = target_sent_data[target_id]
        rec_data = target_rec_data[target_id]

        if not sent_data and not rec_data:
            continue

        p1 = ax.bar(
            list(sent_data.keys()),
            list(sent_data.values()),
            bottom=list(prev_data.values()),
            color=target.color,
        )

        # Add text labels to show which target is which.
        for i in range(len(sent_data.values())):
            ax.text(
                i,
                list(prev_data.values())[i],
                target.name,
                ha='center',
                va='bottom',
                color='black',
            )

        # Add the sent_data values to the prev_data
        for key in sent_data.keys():
            prev_data[key] += sent_data[key]

        p2 = ax.bar(
            list(rec_data.keys()),
            list(rec_data.values()),
            bottom=list(prev_data.values()),
            color=target.color,
            fill=False,
            hatch='//',
            edgecolor=target.color,
        )

        # Add the rec_data values to the prev_data
        for key in rec_data.keys():
            prev_data[key] += rec_data[key]

        count += 1
        if count == 1:
            # Add legend
            ax.legend((p1[0], p2[0]), ('Sent Data', 'Received Data'))

    # Add the labels
    ax.set_ylabel(y_label)

    # Add the x-axis labels
    ax.set_xticks(np.arange(len(sat_names)))
    ax.set_xticklabels(sat_names)

    prefix = f'{prefix}_{filename}' if prefix is not None else filename

    # Save the plot
    if save:
        plots_path = pathlib.Path(__file__).parent.parent / 'plots'
        plt.savefig(plots_path / (prefix + '.png'), dpi=300)

    return fig
