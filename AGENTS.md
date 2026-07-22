# Repository Agent Instructions

## Verification Policy

- Do not add, restore, or maintain unit-test or regression-test files unless the user explicitly requests them.
- Do not treat a regression suite as a routine completion requirement for code changes in this repository.
- For normal development, use lightweight checks: compile the edited Python surface, parse affected configs, import affected modules, and run one relevant real-data smoke workflow only when needed.
- Do not create test scaffolding solely to satisfy generic engineering expectations.
- Keep generated data, model caches, experiment outputs, and smoke artifacts outside Git.
