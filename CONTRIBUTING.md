# Contributing

This repository is primarily a research artifact. Contributions should be small, reviewable, and directly related to correctness or reproducibility.

## Reporting a problem

Open a GitHub issue and include:

- the command that was run;
- the configuration file or minimal parameters;
- the Python and package versions;
- the complete error message or the unexpected metric;
- a minimal synthetic example when third-party data cannot be shared.

Do not attach copyrighted videos, licensed feature files, credentials, private server paths, or unpublished manuscript files.

## Proposing a change

1. Create a branch from `main`.
2. Keep the change focused and document any altered metric or protocol semantics.
3. Add or update tests for behavioral changes.
4. Run `python -m pytest -q`.
5. Open a pull request that explains the scientific and software impact.

Changes to frozen manuscript protocols or derived summaries must also update the relevant checksums and explain why the archived result changed.
