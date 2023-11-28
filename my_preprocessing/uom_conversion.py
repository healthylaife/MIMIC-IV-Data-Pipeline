import pandas as pd
import numpy as np


def drop_wrong_uom(data, cut_off):
    """Drop rows with uncommon units of measurement for each itemid, based on a cut-off frequency."""

    # Create a function to filter each group
    def filter_group(group):
        value_counts = group["valueuom"].value_counts()
        most_frequent_uom = value_counts.idxmax()
        frequency = value_counts.max()

        # Check if the most frequent uom meets the cut-off criteria
        if frequency / len(group) > cut_off:
            return group[group["valueuom"] == most_frequent_uom]
        return group

    # Apply the filter function to each group and concatenate the results
    return (
        data.groupby("itemid", group_keys=False)
        .apply(filter_group)
        .reset_index(drop=True)
    )
