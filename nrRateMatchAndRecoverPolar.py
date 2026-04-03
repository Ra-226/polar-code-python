
import numpy as np
from utils import *


def nrRateMatchPolar(in_, K, E, *varargin):
    # nrRateMatchPolar Polar rate matching
    #   OUT = nrRateMatchPolar(IN,K,E) returns the rate-matched output, OUT,
    #   for a polar encoded input, IN, for an information block length K. The
    #   output is of length E. Coded-bit interleaving is disabled (IBIL is set
    #   to false) for downlink configurations.
    #
    #   OUT = nrRateMatchPolar(...,IBIL) allows the enabling of coded-bit
    #   interleaving by specifying a boolean scalar (IBIL as true). This
    #   setting is used for uplink configurations.
    #
    #   # Example:
    #   # Rate match a polar encoded code block of length 512 to a vector of
    #   # length 864.
    #
    #   N = 2**9            # Polar encoded block length
    #   K = 56             # Number of information bits
    #   E = 864            # Number of rate-matched output bits
    #   iBIL = False       # Interleaving of rate-matched coded bits
    #
    #   in_ = np.random.randint(0, 2, (N, 1))
    #   out = nrRateMatchPolar(in_, K, E, iBIL)
    #
    #   See also nrRateRecoverPolar, nrPolarEncode, nrCRCEncode, nrDCIEncode,
    #   nrBCH.
    #

    nargin = 3 + len(varargin)
    if nargin == 3:
        iBIL = False
    else:
        iBIL = varargin[0]

    # Validate inputs
    validateInputsMatch(in_, K, E, iBIL)

    # Sub-block interleaving, Section 5.4.1.1
    y = subBlockInterleave(in_)

    # Bit selection, Section 5.4.1.2
    N = len(in_)
    outE = np.zeros(E, dtype=in_.dtype)
    if E >= N:
        # Bit repetition
        for k in range(E):
            outE[k] = y[k % N]
    else:
        if K / E <= 7 / 16:
            # puncturing (take from the end)
            outE = y[-E:]
        else:
            # shortening (take from the start)
            outE = y[:E]

    # Interleaving, Section 5.4.1.3
    if iBIL:
        # Specified for uplink only
        out = iBILInterl(outE)
    else:
        # No interleaving
        out = outE

    return out


def nrRateRecoverPolar(in_, K, N, *varargin):
    # nrRateRecoverPolar Polar rate matching recovery
    #   OUT = nrRateRecoverPolar(IN,K,N) returns the rate-recovered output,
    #   OUT, for an input, IN, of length E. The output, OUT, is of length N and
    #   K represents the information block length. Coded-bit interleaving is
    #   disabled (iBIL is set to false) for downlink configurations.
    #
    #   OUT = nrRateRecoverPolar(...,IBIL) allows the enabling of coded-bit
    #   interleaving by specifying a boolean scalar (IBIL as true). This
    #   setting is used for uplink configurations.
    #
    #   # Example:
    #   # Rate match a polar encoded code block of length 512 to a vector of
    #   # length 864 and then recover it.
    #
    #   N = 2**9            # Polar encoded block length
    #   K = 56             # Number of information bits
    #   E = 864            # Number of rate matched output bits
    #   iBIL = False       # Deinterleaving of input bits
    #
    #   in_ = np.random.randint(0, 2, (N, 1))
    #   chIn = nrRateMatchPolar(in_, K, E, iBIL)
    #   out = nrRateRecoverPolar(chIn, K, N, iBIL)
    #   np.array_equal(out, in_)
    #
    #   See also nrRateMatchPolar, nrPolarDecode, nrCRCDecode, nrDCIDecode,
    #   nrBCHDecode.
    #
    

    nargin = 3 + len(varargin)
    if nargin == 3:
        iBIL = False
    else:
        iBIL = varargin[0]

    # Validate inputs
    validateInputsRecover(in_, K, N, iBIL)

    # Channel deinterleaving, Section 5.4.1.3
    if iBIL:
        # Specified for uplink only
        inE = iBILDeinterl(in_)
    else:
        # No deinterleaving
        inE = in_

    # Bit selection, Section 5.4.1.2
    E = len(in_)
    if E >= N:
        # Just the first set output
        outN = inE[:N]
    else:
        if K / E <= 7 / 16:
            # puncturing (put at the end)
            outN = np.zeros(N, dtype=in_.dtype)          # 0s for punctures
            outN[-E:] = inE
        else:
            # shortening (put at the start)
            outN = 1e20 * np.ones(N, dtype=in_.dtype)      # use a large value for 0s
            outN[:E] = inE

    # Sub-block deinterleaving, Section 5.4.1.1
    out = subBlockDeinterleave(outN)

    return out



def validateInputsMatch(in_, K, E, iBIL):
    # Check inputs

    fcnName = 'nrRateMatchPolar'

    # Validate polar-encoded message, length must be a power of two
    if not isinstance(in_, np.ndarray) or in_.ndim != 1 or not np.all(np.logical_or(in_ == 0, in_ == 1)):
        raise ValueError(f'{fcnName}: IN must be a binary column vector')
    N = len(in_)
    if int(np.log2(N)) != np.log2(N):
        raise ValueError('nr5g:nrPolar:InvalidInputRMLength')

    # Validate coded-bit interleaving flag
    if not isinstance(iBIL, bool):
        raise ValueError(f'{fcnName}: IBIL must be a boolean scalar')

    if iBIL:  # for Uplink
        # Validate the information block length which must be greater than
        # or equal to 18 (12+6) and less than N. Also, 25<K<31 is invalid.
        if not (isinstance(K, int) and K >= 18 and K < N):
            raise ValueError(f'{fcnName}: Invalid K')
        if 25 < K < 31:
            raise ValueError('nr5g:nrPolar:UnsupportedKforUL')
    else:  # for Downlink
        # Validate the information block length which must be greater than
        # or equal to 36 (12+24) and less than N
        if not (isinstance(K, int) and K >= 36 and K < N):
            raise ValueError(f'{fcnName}: Invalid K')

    if 18 <= K <= 25:  # for PC-Polar
        nPC = 3
    else:
        nPC = 0

    # Validate rate-matched output length which must be less than or equal
    # to 8192 and greater than K+nPC
    if not (isinstance(E, int) and E > K + nPC and E <= 8192):
        raise ValueError(f'{fcnName}: Invalid E')


def validateInputsRecover(in_, K, N, iBIL):
    # Check inputs

    fcnName = 'nrRateRecoverPolar'

    # Validate log-likelihood ratio value input, length must be less than
    # or equal to 8192
    if not isinstance(in_, np.ndarray) or in_.ndim != 1 or not np.isrealobj(in_):
        raise ValueError(f'{fcnName}: IN must be a real column vector')
    E = len(in_)
    if E > 8192:
        raise ValueError('nr5g:nrPolar:InvalidInputRRLength')

    # Validate coded-bit deinterleaving flag
    if not isinstance(iBIL, bool):
        raise ValueError(f'{fcnName}: iBIL must be a boolean scalar')

    if iBIL:  # for Uplink
        # Validate the information block length which must be greater than
        # or equal to 18 (12+6) and less than E. Also, 25<K<31 is invalid.
        if not (isinstance(K, int) and K >= 18 and K < E):
            raise ValueError(f'{fcnName}: Invalid K')
        if 25 < K < 31:
            raise ValueError('nr5g:nrPolar:UnsupportedKforUL')
    else:  # for Downlink
        # Validate the information block length which must be greater than
        # or equal to 36 (12+24) and less than E
        if not (isinstance(K, int) and K >= 36 and K < E):
            raise ValueError(f'{fcnName}: Invalid K')

    # Validate the polar-encoded output length which must be a power of two
    # and greater than K
    if not (isinstance(N, int) and N > K):
        raise ValueError(f'{fcnName}: Invalid N')
    if int(np.log2(N)) != np.log2(N):
        raise ValueError('nr5g:nrPolar:InvalidN')

def subBlockDeinterleave(in_):
    # Sub-block deinterleaver
    #   OUT = subBlockDeinterleave(IN) returns the sub-block deinterleaved
    #   input.
    #
    #   Reference: TS 38.212, Section 5.4.1.1.

    N = len(in_)
    jn = subblockInterleaveMap(N)
    out = np.zeros(N, dtype=in_.dtype)
    out[jn] = in_

    return out

def iBILDeinterl(in_):
    # Triangular deinterleaver
    #
    #   OUT = iBILDeinterl(IN) performs triangular deinterleaving on the input,
    #   IN, and returns the output, OUT.
    #
    #   Reference: TS 38.212, Section 5.4.1.3.

    # Get T off E
    E = len(in_)
    T = getT(E)

    # Create the table with nulls (filled in row-wise)
    vTab = np.zeros((T, T), dtype=int)
    k = 0
    for i in range(T):
        for j in range(T - i):
            if k < E:
                vTab[i, j] = k
            k += 1

    # Write input to buffer column-wise, respecting vTab
    v = np.inf * np.ones((T, T), dtype=in_.dtype)
    k = 0
    for j in range(T):
        for i in range(T - j):
            if k < E and vTab[i, j] != 0:
                v[i, j] = in_[k]
                k += 1

    # Read output from buffer row-wise
    out = np.zeros(E, dtype=in_.dtype)
    k = 0
    for i in range(T):
        for j in range(T - i):
            if not np.isinf(v[i, j]):
                out[k] = v[i, j]
                k += 1

    return out
