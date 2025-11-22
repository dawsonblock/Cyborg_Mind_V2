"""Sanity test for module imports.

This test simply tries to import the top‑level ``capsule_brain`` package
and a few submodules.  It will fail if there are missing imports or
syntax errors.
"""

def test_imports() -> None:
    import capsule_brain
    import capsule_brain.core
    import capsule_brain.pmm
    import capsule_brain.frnn
    import capsule_brain.workspace
    import capsule_brain.capsules
    import capsule_brain.skills

    # Ensure subpackages are present
    assert hasattr(capsule_brain, "core")
    assert hasattr(capsule_brain, "pmm")