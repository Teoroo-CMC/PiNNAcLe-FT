# -*- coding: utf-8 -*-

import click

from .common import (
    load_opts,
    write_opts,
    shuffle_opts,
    filter_opts,
    subsample_opts,
    load_ds_with_opts,
)


@click.command(name="convert", short_help="convert datasets")
@click.argument("dataset", nargs=-1)
@load_opts
@write_opts
@shuffle_opts
@filter_opts
@subsample_opts
def convert(
    dataset,
    fmt,
    emap,
    # ^ load_opts
    output,
    ofmt,
    # ^ write_opts,
    shuffle,
    seed,
    # ^ shuffle_opts
    filters,
    # ^ filter_opts
    subsample,
    nsample,
    psample,
    sort_key,
    start_idx,
    # ^ subsample_opts
):
    if not dataset:
        return
    ds = load_ds_with_opts(dataset, fmt, emap)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if filters:
        ds = ds.filter(filters)
    if subsample:
        idx, ds = ds.subsample(subsample, nsample, psample, sort_key, start_idx)
        if ofmt == "idx.xyz":
            from ase.io import write

            traj = ds.convert(fmt="ase")
            for i, geo in zip(idx, traj):
                write(f"{i}.xyz", geo)
            return

    ds.convert(output, fmt=ofmt)
