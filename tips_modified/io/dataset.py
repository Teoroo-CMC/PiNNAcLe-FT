# -*- coding: utf-8 -*-

"""This module implements the Dataset class, for data manuputaion in TIPS

"""
import tips
import logging
import numpy as np
from copy import deepcopy

logger = logging.getLogger("tips")


class Dataset:
    def __init__(self, generator=None, indexer=None, meta=None):
        """Dataset should be initialized with a generator, optionally, an
        inedexer

        Args:
           generator (func): function that returns a data generator
           indexer (func): function that returns a single data point given the index
           meta (dic): dataset metadata
        """
        self.meta = meta
        self.indexer = indexer
        if indexer is not None and generator is None:

            def generator():
                for i in range(self.meta["size"]):
                    yield self.indexer(i)

        self.generator = generator

    def __getitem_indexer(self, key):
        if isinstance(key, slice):
            keys = list(range(self.meta["size"]))[key]
            meta = deepcopy(self.meta)
            meta["size"] = len(keys)
            meta["fmt"] = "TIPS sliced"

            def new_indexer(i):
                return self.indexer(keys[i])

            return Dataset(indexer=new_indexer, meta=meta)
        elif isinstance(key, int):
            if key < 0:
                key = key + self.meta["size"]
            if key >= self.meta["size"] or key < 0:
                raise IndexError("Index out of range.")
            return self.indexer(key)
        else:
            raise NotImplementedError(f"Cannot index a dataset with {type(key)}")

    def __getitem_generator(self, key):
        logger.warning("Indexing a generator based dataset, this can be slow.")
        if "size" not in self.meta:
            self.skim()
        size = self.meta["size"]
        if isinstance(key, slice):
            assert (
                key.step is None or key.step > 0
            ), "Cannot slice generator ds with a negative slice."
            keys = list(range(size))[key]
            meta = deepcopy(self.meta)
            meta["size"] = len(keys)
            meta["fmt"] = "TIPS sliced"

            def new_generator():
                iterator = enumerate(self.generator())
                while len(keys) != 0:
                    idx, datum = next(iterator)
                    if idx == keys[0]:
                        del keys[0]
                        yield datum
                    else:
                        assert idx < keys[0], "Unexpected generator output."

            return Dataset(generator=new_generator, meta=meta)
        elif isinstance(key, int):
            if key < 0:
                key = key + self.meta["size"]
            if key >= self.meta["size"] or key < 0:
                raise IndexError("Index out of range.")
            for idx, datum in enumerate(self.generator()):
                if idx == key:
                    return datum
                else:
                    continue
        else:
            raise NotImplementedError(f"Cannot index a dataset with {type(key)}")

    def __getitem__(self, key):
        if self.indexer is None:
            return self.__getitem_generator(key)
        else:
            return self.__getitem_indexer(key)

    def __iter__(self):
        return self.generator()

    def __repr__(self):
        from ase.data import chemical_symbols

        spec_repr = "\n     ".join(
            f"{k}: [{', '.join(['na' if s is None else str(s) for s in v['shape']])}], {v['dtype']}"
            for k, v in self.meta["spec"].items()
        )
        repr = f"""<tips.io.Dataset
 fmt: {self.meta['fmt']}
 size: {self.meta['size'] if 'size' in self.meta else 'Unknown'}
 elem: {', '.join(chemical_symbols[i] for i in sorted(self.meta['elem'])) if 'elem' in self.meta else 'Unknown'}
 spec:
     {spec_repr}>"""
        return repr

    def skim(self, check_elem=True):
        """Scan througn the dataset and build some missing information when possible"""
        count = 0
        if check_elem:
            elem = set()
        for datum in self.generator():
            count += 1
            if check_elem:
                elem = elem.union(datum["elem"])

        if "size" not in self.meta:
            self.meta["size"] = count
            logger.info("Dataset size added to metadata")
        elif self.meta["size"] != count:
            self.meta["size"] = count
            logger.warning("Missing/inconsistent size found, overwriting the metadata")

        if check_elem:
            if "elem" not in self.meta:
                self.meta["elem"] = elem
            elif self.meta["elem"] != elem:
                self.meta["elem"] = elem
                logger.warning("Inconsistent elem set found, overwriting the metadata")

    def convert(self, *args, fmt=None, **kwargs):
        """Convert the dataset to a known format the format, the format writer
        should be one of the registered writers, with the `tips.convertor`
        decorator. See [#custom-readerwriter](./#custom-readerwriter) for an
        example. Args: fmt (str): target format as defined in tips.io.convertors

        Returns:
           converted (any): dependent on the convertor

        """
        convertor = tips.io.convertors[fmt]
        return convertor(self, *args, **kwargs)

    def shuffle(self, seed=0):
        """Shuffle the dataset, support only indexanle datasets only

        Args:
            seed (int): random seed for splitting

        Returns:
            shufled (Dataset): shuffled dataset
        """
        import random

        assert self.indexer is not None
        random.seed(seed)
        new_idx = list(range(self.meta["size"]))
        random.shuffle(new_idx)

        def indexer(i):
            return self.indexer(new_idx[i])

        meta = deepcopy(self.meta)
        meta["fmt"] = "TIPS shuffled"

        return Dataset(indexer=indexer, meta=meta)

    def filter(self, filters):
        """Filters the dataset with a list of filters

        Args:
            filters (list): list of filters

        Returns:
            filtered (Dataset): a TIPS Dataset

        """
        import re

        from .filter import filters2fn

        meta = deepcopy(self.meta)
        meta["fmt"] = "TIPS filtered"
        meta["size"] = None

        filter_fn = filters2fn(filters)

        def generator():
            """New dataset generator that keeps only filter_fn(datum) is True"""
            for datum in self.generator():
                if filter_fn(datum):
                    yield datum

        return Dataset(generator=generator, meta=meta)

    def subsample(self, strategy, nsample=None, psample=None, sort_key="force_std", start_idx=None):
        """Subsample a dataet according to certain variables

        Args:
            strategy (str): 'head', 'tail', 'uniform' or 'sorted'
            nsample (int): number of samples to take
            psample (float): percentile of samples to take
            sort_key (str): key used to sort the data
            start_idx (int): the start index for sampling, added by zhan-yun zhang 20260418

        Returns:
            idx (int): inidices of the selected samples
            subsampled (Dataset): a TIPS Dataset
        """
        import math
        import numpy as np

        if "size" not in self.meta or self.meta["size"] is None:
            self.skim()
        size = self.meta["size"]
        # when using PiNNAcLe-FT, the nsample is equal to size in gen0
        if nsample is not None and nsample == size:
            start_idx = 0
        elif start_idx is None:
            start_idx = 0
        assert (nsample is None) != (
            psample is None
        ), "Must specify only one of nsample or psample."
        assert (psample is None or start_idx == 0
        ), f"The non-zero start_idx is incompatible with the psample."
        valid_size = size - start_idx
        if psample is not None:
            nsample = int(valid_size / 100.0 * psample)
        assert (
            nsample <= valid_size
        ), f"Requested sample {nsample} larger than dataset size {valid_size}."
        if strategy == "head":
            idx = np.arange(start_idx, start_idx+nsample, dtype=int)
            dataset = self[slice(start_idx, start_idx+nsample)]
        elif strategy == "tail":
            assert (size - nsample >= start_idx
            ), "The starting idx is below the start_idx."
            idx = np.arange(size - nsample, size, dtype=int)
            dataset = self[slice(size - nsample, size)]
        elif strategy == "uniform":
            step = max(1, valid_size // nsample)
            stop_idx = start_idx + step * nsample
            idx = np.array(list(range(start_idx, stop_idx, step)))
            dataset = self[slice(start_idx, stop_idx, step)]
        elif strategy == "sorted":
            if sort_key.startswith("-"):
                key = sort_key[1:]
                sign = - 1
            else:
                key = sort_key
                sign = 1
            tosort = [sign * np.max(datum[key]) for datum in self]
            idx = np.argsort(tosort)[::-1][:nsample]
            idx = idx[idx >= start_idx]

            def indexer(i):
                return self.indexer(idx[i])

            meta = deepcopy(self.meta)
            meta["fmt"] = "TIPS subsampled"
            meta["size"] = nsample
            dataset = Dataset(indexer=indexer, meta=meta)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        return idx, dataset

    def join(self, ds):
        """Joins two datasets

        Args:
            ds (Dataset): a tips dataset object

        Returns:
            joined (Dataset): joined dataset
        """

        # check the compatibility of two datsets
        assert set(self.meta["spec"].keys()) == set(self.meta["spec"].keys())
        meta = deepcopy(self.meta)
        meta["fmt"] = "TIPS joined"
        meta["size"] = self.meta["size"] + ds.meta["size"]
        if ("elem" in self.meta) and ("elem" in ds.meta):
            meta["elem"] = self.meta["elem"].union(ds.meta["elem"])
        keys = set(meta["spec"].keys()).intersection(set(ds.meta["spec"].keys()))
        meta["spec"] = {k: v for k, v in meta["spec"].items() if k in keys}
        if self.indexer is not None and ds.indexer is not None:

            def indexer(i):
                if i < self.meta["size"]:
                    datum = self.indexer(i)
                else:
                    datum = ds.indexer(i - self.meta["size"])
                return {k: v for k, v in datum.items() if k in keys}

            return Dataset(indexer=indexer, meta=meta)

    def map_elems(self, emap):
        """Maps the elements of a dataset according to some typing rule

        Args:
            emap (dict or str): a dict or a LAMMPS data file

        Returns:
            mapped (Dataset): mapped dataset
        """
        import numpy as np
        from ase.data import atomic_masses

        if isinstance(emap, dict):
            edict = emap

        edict = {}
        with open(emap) as f:
            while True:
                l = f.readline()
                if l.startswith("Masses"):
                    break
            f.readline()
            while True:
                line = f.readline().split()
                if len(line) == 0:
                    break
                else:
                    idx = int(line[0])
                    mass = float(line[1])
                    edict[idx] = max(1, np.argmin(np.abs(atomic_masses - mass)))

        def mapper(datum):
            datum = datum.copy()
            datum["elem"] = [edict[i] for i in datum["elem"]]
            return datum

        meta = deepcopy(self.meta)
        meta["fmt"] = "TIPS mapped"
        meta["elem"] = set([edict[i] for i in meta["elem"]])
        if self.indexer is None:

            def generator():
                for datum in self.generator():
                    yield mapper(datum)

            return Dataset(generator=generator, meta=meta)
        else:

            def indexer(i):
                return mapper(self.indexer(i))

            return Dataset(indexer=indexer, meta=meta)
