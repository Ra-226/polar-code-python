# polar-code-python

English | [简体中文](README.zh.md)

Python implementation of core 5G NR Polar coding components, converted and
adapted from MATLAB-style workflows for experimentation, learning, and
simulation.

This repository focuses on CRC processing, Polar encoding/decoding, rate
matching/recovery, and a couple of end-to-end simulation scripts for error-rate
analysis.

## Overview

The codebase contains a small set of function-oriented modules implementing key
steps in the 5G NR Polar coding chain:

- CRC encoding and decoding
- Polar code construction
- Polar encoding
- CRC-aided successive-cancellation list decoding
- Polar rate matching and rate recovery
- Simulation scripts for BER/FER exploration

The implementation style intentionally stays close to the original algorithmic
structure, so the code is easier to compare against textbook or standard-based
descriptions of Polar coding.

## Standards Reference

The implementation and comments in this repository reference 3GPP NR channel
coding behavior, especially:

- `3GPP TS 38.212, "3rd Generation Partnership Project; Technical
  Specification Group Radio Access Network; NR; Multiplexing and channel
  coding"`

In particular, the source code references sections such as:

- Section 5.1 for CRC processing
- Section 5.3.1 for Polar coding
- Section 5.3.1.1 for input interleaving
- Section 5.3.1.2 for Polar code construction / reliability sequence
- Section 5.4.1.1 for sub-block interleaving
- Section 5.4.1.2 for bit selection in rate matching
- Section 5.4.1.3 for coded-bit interleaving

This project is best understood as an educational and experimental Python
implementation aligned with those references, not an officially certified 3GPP
stack.

## Repository Structure

- `nrCRC.py` - CRC encode/decode utilities for NR-related polynomials
- `utils.py` - helper functions such as reliability sequence, interleavers, and
  code construction helpers
- `nrPolarEncode.py` - Polar encoder
- `nrPolarDecode.py` - CRC-aided Polar list decoder
- `nrRateMatchAndRecoverPolar.py` - rate matching and rate recovery for Polar
  codes
- `detailedPerformanceAnalysis.py` - analysis-oriented simulation script
- `polarAudioWatermarkSim.py` - simulation script for watermark/channel
  robustness experiments
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+ recommended
- `numpy`
- `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. CRC example

```python
import numpy as np
from nrCRC import nrCRCEncode, nrCRCDecode

msg = np.array([1, 0, 1, 1, 0, 1], dtype=int)
encoded = nrCRCEncode(msg, '11')
decoded, err = nrCRCDecode(encoded, '11')

print(encoded.shape)
print(decoded.flatten())
print(err)
```

### 2. Polar encode / decode example

```python
import numpy as np
from nrCRC import nrCRCEncode
from nrPolarEncode import nrPolarEncode
from nrPolarDecode import nrPolarDecode

K_info = 64
crc_len = 11
E = 128
L = 8
nMax = 10
iIL = False

info = np.random.randint(0, 2, K_info)
msg = nrCRCEncode(info, '11').flatten()
enc = nrPolarEncode(msg, E, nMax, iIL).astype(float)

# Perfect-channel LLRs
llr = (1 - 2 * enc) * 8.0
dec = nrPolarDecode(llr, len(msg), E, L, nMax, iIL, crc_len)

print(dec[:K_info])
```

### 3. Run the included scripts

```bash
python detailedPerformanceAnalysis.py
python polarAudioWatermarkSim.py
```

## Module Notes

### `nrCRC.py`

Provides:

- `nrCRCEncode`
- `nrCRCDecode`

Supported CRC identifiers include:

- `'6'`
- `'11'`
- `'16'`
- `'24A'`
- `'24B'`
- `'24C'`

Important distinction:

- `nrCRCEncode` / `nrCRCDecode` support the full set above.
- The Polar workflow does not use all of them interchangeably.
- `nrPolarDecode(..., crcLen)` accepts CRC lengths `6`, `11`, or `24`.
- In the current Polar decoder implementation, `crcLen=24` maps to the NR
  downlink CRC polynomial `'24C'`.
- So `'16'`, `'24A'`, and `'24B'` are available in the standalone CRC utility
  module, but they are not general `nrPolarDecode` options.

### `nrPolarEncode.py`

Main entry point:

- `nrPolarEncode(inp, E)` for default downlink-style settings
- `nrPolarEncode(inp, E, nMax, iIL)` for explicit control

Typical parameter conventions:

- `inp`: 1D binary NumPy array containing message bits including CRC bits
- `inp` is expected to already include the CRC bits produced before Polar
  encoding
- `E`: rate-matched output length
- `nMax`: usually `9` or `10`
- `iIL`: input interleaving flag

### `nrPolarDecode.py`

Main entry point:

- `nrPolarDecode(inp, K, E, L)`
- `nrPolarDecode(inp, K, E, L, nMax, iIL, crcLen)`

Typical parameter conventions:

- `inp`: 1D float array of LLRs
- `K`: total number of Polar message bits, including appended CRC bits
- `E`: rate-matched output length
- `L`: list size for SCL decoding

For example, if the payload has `K_info=64` bits and CRC length is `11`, then
the Polar encoder/decoder should use `K=75`.

### `nrRateMatchAndRecoverPolar.py`

Provides:

- `nrRateMatchPolar`
- `nrRateRecoverPolar`

These functions implement the interleaving and bit-selection stages used around
Polar coding in NR-style workflows.

## Validation Commands

There is currently no formal test suite in the repository, but these commands
are useful for basic validation.

Syntax-check all core files:

```bash
python -m py_compile utils.py nrCRC.py nrPolarEncode.py nrPolarDecode.py nrRateMatchAndRecoverPolar.py detailedPerformanceAnalysis.py polarAudioWatermarkSim.py
```

Import smoke test:

```bash
python -c "import nrCRC, nrPolarEncode, nrPolarDecode, nrRateMatchAndRecoverPolar, utils"
```

Single-file syntax check:

```bash
python -m py_compile nrPolarDecode.py
```

Ad hoc CRC sanity check:

```bash
python -c "import numpy as np; from nrCRC import nrCRCEncode; print(nrCRCEncode(np.array([1,0,1]), '11').shape)"
```

## Current Limitations

- There is no packaged install layout yet.
- There is no automated `pytest` suite yet.
- The repository is primarily aimed at experimentation and reference, not at
  high-performance production deployment.
- Some scripts and APIs still reflect MATLAB-style parameter conventions and
  translated control flow for easier standard/code comparison.

## Recommended Workflow

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Start with CRC and encoder/decoder smoke tests.
4. Run `detailedPerformanceAnalysis.py` for basic behavior inspection.
5. Use `polarAudioWatermarkSim.py` for broader channel robustness experiments.

## License

This project is released under the MIT License. See `LICENSE` for details.
