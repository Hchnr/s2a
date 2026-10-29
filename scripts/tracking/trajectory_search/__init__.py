"""Registered full-trajectory search producers and independent audits."""

from s2a.infrastructure.legacy_script_imports import verify_legacy_modules

# Inspect only already loaded, explicitly retained legacy modules. No imports or
# experiment work are triggered by this package boundary.
_legacy_import_receipt = verify_legacy_modules()
