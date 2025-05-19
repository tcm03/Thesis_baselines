from models.utils import log_rank0
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Sampler

# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Args:
        indices: list of indices
        lengths: list of lengths of those indices
        num_chunks: number of chunks to split the indices into

    Returns:
        list of chunks of length num_chunks, each chunk is a list of indices

    Description:
        Split a list of indices into `chunks` chunks of roughly equal lengths.

        Goal: Given a flat list of sample indices plus their “lengths,” split 
        them into num_chunks groups so that each group has (roughly) the same 
        total length and (roughly) the same number of indices.

        Why: When you later divide work across world_size workers, you don’t want 
        one rank to get all the long samples and another rank only shorts. This 
        evens out both sample count and aggregate length.
    """

    if len(indices) % num_chunks != 0:
        # @tcm: If they don’t divide evenly by count, just round-robin
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):
    """
    Args:
        lengths: list of modality lengths of each video (negative)
        batch_size: per-device batch size
        world_size: number of devices in distributed group
        generator: random number generator
        merge: not used

    Returns:
        list of reordered indices of videos in the original dataset, flattened from
            megabatches: list of megabatchs (whole dataset split into megabatches randomly)
            megabatch: list of batches (of length world_size, split by split_to_even_chunks function)
            batch: list of indices

    Description:
        Produce a single flat list of all dataset indices, such that:
        1. The data are arranged in megabatches of size world_size * batch_size.
        2. Within each megabatch, indices are sorted by descending length in negative values
         (ascending lengths in real values: -1, -3, -7, -12, ...).
        3. Each megabatch is then split into world_size even-length chunks (via split_to_even_chunks), and these 
        chunks are interleaved to form the final order.
    """
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    # for mb, megabatch in enumerate(megabatches):
    #     log_rank0(f"megabatches[{mb}] = {megabatch}")
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]
    # log_rank0(f"get_length_grouped_indices returns: {[i for megabatch in megabatches for batch in megabatch for i in batch]}")
    # @tcm: return [mb0_r0_id0, mb0_r0_id1, ..., mb0_r1_id0, mb0_r1_id1, ..., mb1_r0_id0, ...]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_modality_length_grouped_indices(
    lengths, batch_size, world_size, generator=None
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        # @tcm: If everything is video (positive) or everything is text (negative), just fall back.
        return get_length_grouped_indices(
            lengths, batch_size, world_size, generator=generator
        )
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(
            mm_lengths, batch_size, world_size, generator=None
        )
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(
            lang_lengths, batch_size, world_size, generator=None
        )
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths # @tcm: negative total number of words within the conversation of each sample
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return int(len(self.lengths) // self.world_size) # @tcm: number of samples (video/ json item) in the dataset

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        rank = int(os.environ.get("RANK", 0))
        expected_split_size = len(indices) // self.world_size
        # mb0_r0_i0, mb0_r0_i1, mb0_r1_i0, mb0_r1_i1, mb1_r0_i0, mb1_r0_i1, mb1_r1_i0, mb1_r1_i1, mb2_r0_i0, mb2_r0_i1, mb2_r1_i0
        rank_indices = []
        for i in range(rank * self.batch_size, len(indices), self.batch_size * self.world_size):
            rank_indices.extend(indices[i:i+self.batch_size])
        rank_indices = rank_indices[:expected_split_size]
        return iter(rank_indices)