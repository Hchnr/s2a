"""The recovery audit must preserve prior failure cost before publication."""

import pytest

from scripts.simulation.runtime import audit_bowl_pad_alignment_v2 as audit


def test_recovery_publication_rejects_wrong_result_or_path():
    with pytest.raises(ValueError, match="registered recovered result"):
        audit.recovered_publication(audit.OUTPUT, {"source_ref": audit.FAILURE, "audited_points": 97})
    with pytest.raises(ValueError, match="registered recovered result"):
        audit.recovered_publication(
            audit.OUTPUT.parent / "wrong.json", {"source_ref": audit.SOURCE, "audited_points": 97}
        )
