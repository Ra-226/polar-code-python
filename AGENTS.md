# AGENTS.md

This file gives agentic coding assistants the minimum project-specific context
needed to work safely in this repository.

## Project Snapshot

- Repository type: small Python codebase for 5G NR Polar coding utilities and
  simulation scripts.
- Main modules: `nrCRC.py`, `nrPolarEncode.py`, `nrPolarDecode.py`,
  `nrRateMatchAndRecoverPolar.py`, `utils.py`.
- Main scripts: `detailedPerformanceAnalysis.py`, `polarAudioWatermarkSim.py`.
- There is no package metadata (`pyproject.toml`, `setup.py`, `requirements.txt`,
  `pytest.ini`, `tox.ini`) in the repo root.
- There is no existing `AGENTS.md` to preserve.

## Agent Rules Sources

- Cursor rules: none found in `.cursor/rules/`.
- Cursor root rules: no `.cursorrules` file found.
- Copilot rules: no `.github/copilot-instructions.md` found.
- Because no external agent rules exist, follow this file plus the existing code
  patterns in the touched module.

## Environment Notes

- Platform in this workspace is Windows.
- The repository is a git repo.
- In the current environment, `python` and `py -3` were not available, so do not
  assume local command execution will succeed without first checking.
- Prefer commands that are cross-platform and do not rely on Unix shell tools.

## Build / Run / Validation Commands

This repo does not have a formal build system. Treat "build" as import or syntax
validation plus script execution.

### Basic commands

- Syntax-check all core files:
  `python -m py_compile utils.py nrCRC.py nrPolarEncode.py nrPolarDecode.py nrRateMatchAndRecoverPolar.py detailedPerformanceAnalysis.py polarAudioWatermarkSim.py`
- Run the main analysis script:
  `python detailedPerformanceAnalysis.py`
- Run the watermark simulation script:
  `python polarAudioWatermarkSim.py`
- Run a quick import smoke test:
  `python -c "import nrCRC, nrPolarEncode, nrPolarDecode, nrRateMatchAndRecoverPolar, utils"`

### Single-file validation

- Syntax-check one file:
  `python -m py_compile nrPolarDecode.py`
- Import one module only:
  `python -c "import nrCRC"`
- Run one ad hoc function check:
  `python -c "import numpy as np; from nrCRC import nrCRCEncode; print(nrCRCEncode(np.array([1,0,1]), '11').shape)"`

### Tests

- No automated test suite is present today.
- No `tests/` directory, `pytest` config, or `unittest` suite was found.
- If you add tests, prefer `pytest` and place them under `tests/`.
- Recommended full test command once tests exist:
  `python -m pytest`
- Recommended single-test file command once tests exist:
  `python -m pytest tests/test_nr_crc.py`
- Recommended single-test function command once tests exist:
  `python -m pytest tests/test_nr_crc.py -k encode_roundtrip`
- Prefer deterministic tests with fixed random seeds.

### Lint / Format

- No formatter or linter config is checked in.
- Before introducing a tool such as `ruff` or `black`, check whether the user
  wants repo-wide formatting changes.
- If linting is needed for new tests or modules, prefer targeted commands:
  `ruff check nrCRC.py`
  `black nrCRC.py`
- Do not assume these tools are installed.

## Code Organization

- The repository is function-oriented; there are no classes.
- Most functions operate on NumPy arrays and expect 1D vectors.
- Several files are MATLAB-to-Python translations; preserve algorithmic intent
  over stylistic modernization unless the user requests refactoring.
- Large literal tables and reliability sequences in `utils.py` are part of the
  algorithm; avoid reformatting them gratuitously.

## Import Conventions

- Prefer explicit imports over wildcard imports in new code.
- Existing code uses both local imports (`from nrCRC import ...`) and package-like
  imports (`from TAF.ecc.polar...`); preserve the import style already used in the
  file you are editing unless you are intentionally normalizing the module path.
- Do not introduce circular imports.
- Keep standard library imports first, third-party imports next, local imports
  last.
- Current third-party dependency is NumPy; `matplotlib` is only used by
  `detailedPerformanceAnalysis.py`.

## Formatting Guidelines

- Follow PEP 8 spacing and indentation (4 spaces, no tabs).
- Keep functions separated by blank lines.
- Prefer line lengths near 88-100 characters when practical, but prioritize
  readability of mathematical expressions and lookup tables.
- Keep array math readable; split long expressions instead of compressing them.
- Preserve existing comment blocks when they document 3GPP behavior or MATLAB
  provenance.
- Avoid adding decorative comments.

## Types and Data Shapes

- Prefer NumPy arrays for bit vectors and LLR vectors.
- Preserve the current convention that binary vectors are usually integer arrays
  containing only `0` and `1`.
- Use floating-point arrays for LLR or channel-confidence values.
- Flatten column vectors to 1D when required by downstream code.
- When adding validation, check both dtype expectations and dimensionality.
- Be explicit about shape changes, especially when a function may receive `(K,)`
  or `(K, 1)` inputs.

## Naming Conventions

- Match the surrounding file, even if the naming is not idiomatic Python.
- Existing API names intentionally mirror MATLAB / 3GPP names:
  `nrPolarEncode`, `nrPolarDecode`, `nrCRCEncode`, `subBlockInterleave`, etc.
- Use `camelCase` for new public helpers inside these translated modules if they
  conceptually extend the existing API.
- Use short mathematical variable names only when they match the coding-theory
  notation (`N`, `K`, `E`, `L`, `F`, `qPC`, `llr`).
- Use descriptive names for new high-level script variables.

## Error Handling

- The current code raises `ValueError` for invalid user inputs; keep that pattern.
- Error messages are short and function-prefixed; maintain that style.
- Validate early, before heavy numerical work.
- When a function relies on binary inputs, verify both shape and value domain.
- Avoid silently coercing invalid values except for already-established behavior
  such as flattening `(K,1)` arrays.

## Numerical / Algorithmic Practices

- Preserve deterministic behavior when the script already sets a random seed.
- Do not change decoding logic, frozen-bit construction, or CRC behavior without
  understanding the 3GPP / MATLAB mapping first.
- Be careful with 0-based versus 1-based indexing when translating or editing
  algorithms that came from MATLAB.
- Prefer clarity over micro-optimization unless performance is the task.
- For performance work, benchmark representative vector sizes before rewriting
  loops into vectorized code.

## Editing Guidance

- Make the smallest safe change that solves the task.
- Avoid repo-wide style rewrites; files contain mixed conventions and large data
  tables.
- If you touch import paths, verify whether the file is meant to run as a local
  script or as part of a larger `TAF.ecc.polar` package.
- Keep scripts runnable from the repository root unless the user asks for a
  packaging refactor.
- Add tests for bug fixes or behavior changes when a test harness is introduced.

## Good Agent Defaults

- Read the whole target file before editing; many functions depend on shape and
  indexing conventions established earlier in the file.
- When fixing bugs, prefer a narrow regression test or at least a reproducible
  one-liner command.
- Mention command limitations if the environment lacks Python.
- If a requested change would normalize the entire package structure, ask first;
  that would be a broader refactor than a typical bug fix.
