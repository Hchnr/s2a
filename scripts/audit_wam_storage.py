"""Check logical tensor coverage and files referenced by frozen DCP metadata."""

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    out = root / "checkpoint_storage.json"
    if out.exists():
        raise FileExistsError(out)
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    rows = []
    for checkpoint in c["checkpoints"]:
        directory = Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model"
        metadata = FileSystemReader(directory).read_metadata()
        tensors = []
        for name, item in metadata.state_dict_metadata.items():
            if not isinstance(item, TensorStorageMetadata):
                raise ValueError("unexpected non-tensor model checkpoint entry")
            volume = math.prod(item.size)
            if sum(math.prod(chunk.sizes) for chunk in item.chunks) != volume:
                raise ValueError(f"incomplete logical tensor coverage: {name}")
            for chunk in item.chunks:
                if len(chunk.sizes) != len(item.size) or len(chunk.offsets) != len(item.size):
                    raise ValueError("chunk dimensionality differs")
                if any(
                    o < 0 or s < 0 or o + s > d for o, s, d in zip(chunk.offsets, chunk.sizes, item.size, strict=True)
                ):
                    raise ValueError("chunk extends outside its logical tensor")
            for a, b in itertools.combinations(item.chunks, 2):
                if (
                    math.prod(a.sizes)
                    and math.prod(b.sizes)
                    and all(
                        max(x, y) < min(x + sx, y + sy)
                        for x, sx, y, sy in zip(a.offsets, a.sizes, b.offsets, b.sizes, strict=True)
                    )
                ):
                    raise ValueError(f"overlapping checkpoint chunks: {name}")
            tensors.append(
                dict(
                    name=name,
                    shape=list(item.size),
                    dtype=str(item.properties.dtype),
                    chunks=len(item.chunks),
                    elements=volume,
                )
            )
        locations = Counter()
        for index, storage in metadata.storage_data.items():
            path = directory / storage.relative_path
            if path.parent.resolve() != directory.resolve() or not path.is_file():
                raise ValueError("checkpoint references a missing or unexpected shard path")
            if storage.offset < 0 or storage.length <= 0 or storage.offset + storage.length > path.stat().st_size:
                raise ValueError("checkpoint storage interval exceeds its shard")
            locations[(index.fqn, tuple(index.offset))] += 1
        expected = {
            (name, tuple(chunk.offsets)) for name, item in metadata.state_dict_metadata.items() for chunk in item.chunks
        }
        if set(locations) != expected or any(count != 1 for count in locations.values()):
            raise ValueError("logical chunks do not map one-to-one to checkpoint storage")
        referenced = {value.relative_path for value in metadata.storage_data.values()}
        all_files = {path.name for path in directory.glob("*.distcp")}
        unused = all_files - referenced
        row = dict(
            checkpoint=checkpoint["name"],
            metadata_sha256=sha256(directory / ".metadata"),
            logical_tensors=len(tensors),
            logical_elements=sum(t["elements"] for t in tensors),
            chunks=len(expected),
            referenced_files={name: (directory / name).stat().st_size for name in sorted(referenced)},
            unreferenced_files={name: (directory / name).stat().st_size for name in sorted(unused)},
            coverage_complete_nonoverlapping=True,
            all_storage_intervals_within_files=True,
            tensors=tensors,
        )
        rows.append(row)
        print(
            json.dumps(
                dict(
                    checkpoint=row["checkpoint"],
                    tensors=row["logical_tensors"],
                    referenced=len(referenced),
                    unused=len(unused),
                    unused_bytes=sum(row["unreferenced_files"].values()),
                )
            ),
            flush=True,
        )
    write_json(
        out,
        dict(
            code=code_fingerprint(),
            checkpoints=rows,
            interpretation="DCP reads metadata-referenced shards. This checks complete storage coverage, not historical optimizer correctness or numerical quality of learned parameters.",
        ),
    )


if __name__ == "__main__":
    main()
