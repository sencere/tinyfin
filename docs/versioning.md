# tinyfin versioning and deprecation policy

## Versioning
- Use semantic versioning: MAJOR.MINOR.PATCH.
- MAJOR: breaking API or behavior changes.
- MINOR: new features, APIs, or performance improvements that preserve compatibility.
- PATCH: bug fixes, performance fixes, or docs/test updates with no API changes.

## Stability tiers
- **Stable**: default APIs in `tinyfin` and documented modules.
- **Experimental**: explicitly flagged (e.g., `TINYFIN_ENABLE_HIGHER_ORDER`, IR helpers, graph tooling).
- **Internal**: underscored helpers or C internals; no compatibility guarantees.

## Deprecation process
1. Mark deprecated in docs and release notes, including replacement guidance.
2. Emit a runtime warning for deprecated Python APIs for at least one MINOR release.
3. Remove only in a subsequent MAJOR release (or after two MINORs if the project is <1.0).

## Compatibility rules
- Keep Tensor shape/dtype semantics backward compatible within a MAJOR version.
- Serialization formats maintain backward compatibility across MINOR versions; breaking changes require a MAJOR bump and a migration note.
- Experimental features may change or be removed without deprecation, but should be documented as such.

## Version metadata
- `tinyfin.__version__` (Python) and a C header define the current version.
- Release notes list breaking changes, deprecations, and migration steps.
