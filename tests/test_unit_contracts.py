"""Legacy architecture test path; canonical checks live in tests/architecture."""

from tests.architecture import test_dependencies, test_imports

# Re-export module names only so default collection does not run checks twice.
__all__ = ["test_dependencies", "test_imports"]
