# PRD-06: GitHub Actions build, test and Pages deploy

## Requirements
1. Workflow `.github/workflows/pages.yml` on push to `main`:
   - fetch and verify wasi-sdk (PRD-01) with a cache keyed by version;
   - configure and build the wasm core (PRD-02), transpile (PRD-03);
   - run the Node smoke test;
   - assemble `site/` containing the page, transpiled files and the stick-figure assets;
   - deploy `site/` to the `gh-pages` branch via `actions/deploy-pages`.
2. Model files are not committed and not built into the site (PRD-04).
3. Commits made by automation and by the owner are attributed to `lanmower` only.
4. Pin all action versions by commit SHA.

## Acceptance
- A push to `main` produces a green run and a Pages URL that serves the stick figure page.
- The deployed page is reachable and loads the transpiled component.

## Risks
- Pages must be enabled for the repository with source set to GitHub Actions.
- Runner memory (7 GB) is enough for build but the model is not built in CI.
