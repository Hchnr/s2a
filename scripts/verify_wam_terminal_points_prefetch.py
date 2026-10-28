"""Bounded parallel reads; retain the original ordered compact math audit."""

from concurrent.futures import ThreadPoolExecutor

from verify_wam_clean_training_capture import Evidence, require
from verify_wam_terminal_development_points import audit_points as ordered_audit


class RequestSnapshot:
    """Read one request with private evidence maps, then merge on the main thread."""

    def __init__(self, evidence, ledger, directory):
        require(len(ledger["points"]) == ledger["nodes"] == 35, "Full35 request only")

        def load(item):
            index, ref = item
            local = Evidence()
            record = local.json(ref)
            arrays = local.arrays(record["arrays_ref"], directory / f"compact/step{index:02d}.npz")
            return ref, record, arrays, local.inputs

        self.evidence = evidence
        self.records, self.archives = {}, {}
        # At most one request is retained; completed requests release their arrays.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for ref, record, arrays, inputs in pool.map(load, enumerate(ledger["points"])):
                for path, digest in inputs.items():
                    require(evidence.inputs.get(path, digest) == digest, "Conflicting consumed SHA")
                    evidence.inputs[path] = digest
                self.records[(ref["path"], ref["sha256"])] = record
                array_ref = record["arrays_ref"]
                self.archives[(array_ref["path"], array_ref["sha256"])] = arrays

    def json(self, ref):
        return self.records[(ref["path"], ref["sha256"])]

    def arrays(self, ref, expected_path):
        # Exact paths were checked by the private Evidence loader; the original
        # ordered audit supplies these same expected paths again.
        require(str(expected_path.resolve()) == ref["path"], "Snapshot archive path differs")
        return self.archives[(ref["path"], ref["sha256"])]


def audit_points(evidence, ledger, header, directory, steps, **kwargs):
    snapshot = RequestSnapshot(evidence, ledger, directory)
    return ordered_audit(snapshot, ledger, header, directory, steps, **kwargs)
