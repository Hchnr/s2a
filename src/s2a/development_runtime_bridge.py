"""051 metadata source adapter; caller must supply an independently admitted repair.

Original preparation/registration and mathematical functions remain untouched.
These helpers do not authorize a model or create simulator environments.
"""

from pathlib import Path

from s2a.durable_stage_supervisor import check_sources, reference, require


def extend_sources(original, repair):
    """Keep every original source byte and add the separately admitted closure."""
    require(isinstance(original, dict) and original, "original source closure missing")
    require(isinstance(repair, dict) and repair, "repair source closure missing")
    merged = dict(original)
    for path, digest in repair.items():
        require(path not in merged or merged[path] == digest, "repair changed an original source")
        merged[path] = digest
    check_sources(merged)
    return merged


def original_requirements(required_sources, qualification):
    """Map finite new entrypoints to their byte-checked original admission basis."""
    pairs = qualification["generated_to_original"]
    sources = qualification["source_sha256"]
    original = []
    for required in required_sources:
        path = str(Path(required).resolve())
        require(sources.get(path) == reference(path)["sha256"], "required actual entrypoint is not repair-frozen")
        if path in pairs:
            basis = pairs[path]
            require(reference(basis["path"]) == basis, "frozen original entrypoint changed")
            require(sources.get(basis["path"]) == basis["sha256"], "original basis omitted from repair closure")
            original.append(Path(basis["path"]))
        else:
            original.append(Path(path))
    require(original, "actual entrypoint requirements missing")
    return original


def admit_runtime(runtime, prepared_ref, registration_ref, *, required_sources, qualification):
    """Run the original proof first, then add source refs without replacing its JSON."""
    require(
        prepared_ref == qualification["prepared_ref"]
        and registration_ref == qualification["execution_registration_ref"],
        "repair substituted original041 identity",
    )
    required = original_requirements(required_sources, qualification)
    prepared, registration, sources = runtime.admit_execution(prepared_ref, registration_ref, required_sources=required)
    for actual in required:
        require(sources.get(str(actual)) == reference(actual)["sha256"], "original runtime did not freeze actual basis")
    merged = extend_sources(sources, qualification["source_sha256"])
    return prepared, registration, merged
