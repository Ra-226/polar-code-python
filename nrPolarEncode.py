import numpy as np
from TAF.ecc.polar.utils import *

def nrPolarEncode(inp, E, *varargin):
    # nrPolarEncode Polar encoding
    #   ENC = nrPolarEncode(IN,E) returns the polar encoded output for an input
    #   message, IN, and rate-matched output length, E, for a downlink
    #   configuration. IN is a column vector of length K, including the CRC
    #   bits, as applicable. ENC is the output column vector of length N. For
    #   downlink, nMax = 9 and iIL = True. The function includes the
    #   determination of the frozen bits, input interleaving and encoding as
    #   per TS 38.212 Section 5.3.1.
    #
    #   ENC = nrPolarEncode(IN,E,NMAX,IIL) encodes the input using the
    #   specified NMAX (an integer value of 9 or 10) and IIL (a boolean scalar)
    #   parameters. The allowed value sets of {9,True} and {10,False} for
    #   {NMAX,IIL} apply for downlink and uplink configurations respectively.
    #
    #   # Example 1:
    #   # Polar-encode a block of data
    #
    #   K = 132            # Message length
    #   E = 256            # Rate matched output length
    #   msg = np.random.randint(0, 2, (K, 1))                # Generate random message
    #   enc = nrPolarEncode(msg, E)            # Polar encode
    #   enc.shape[0]
    #
    #   # Example 2:
    #   # Transmit polar-encoded block of data and decode using
    #   # successive-cancellation list decoder.
    #
    #   K = 132            # Message length
    #   E = 256            # Rate matched output length
    #   nVar = 1.0         # Noise variance
    #   L = 8              # Decoding List length
    #
    #   # Simulate a frame
    #   msg    = np.random.randint(0, 2, (K, 1))                      # Generate random message
    #   enc    = nrPolarEncode(msg, E)                  # Polar encode
    #   # Further steps would involve modulation, channel, demodulation, and decoding
    #
    #   See also nrPolarDecode, nrCRCEncode, nrRateMatchPolar, nrDCIEncode,
    #   nrBCH, nrUCIEncode.
    #
    

    nargin = 2 + len(varargin)

    # Parse inputs
    if nargin == 2:
        # Downlink params
        # nrPolarEncode(inp, E)
        nMax = 9       # maximum n value for N
        iIL = True     # input interleaving
    elif nargin == 3:
        raise ValueError('nr5g:nrPolar:InvalidNumInputs')
    else:
        # nrPolarEncode(inp, E, nMax, iIL)
        nMax = varargin[0]
        iIL = varargin[1]

    # Validate inputs
    validateInputs(inp, E, nMax, iIL)

    # Input is a single code block and assumes CRC bits are included
    K = len(inp)

    # Interleave input, if specified
    if iIL:
        pi = interleaveMap(K)
        inIntr = inp[pi]
    else:
        inIntr = inp

    inIntr = inIntr.flatten()
    # Get frozen bit indices and parity-check bit locations
    F, qPC, _ = construct(K, E, nMax)
    N = len(F)
    nPC = len(qPC)

    # Generate u
    u = np.zeros(N, dtype=int)     # integers
    if nPC > 0:
        # Parity-Check Polar (PC-Polar)
        y0 = 0
        y1 = 0
        y2 = 0
        y3 = 0
        y4 = 0
        k = 0
        for idx in range(N):
            yt = y0
            y0 = y1
            y1 = y2
            y2 = y3
            y3 = y4
            y4 = yt
            if F[idx]:   # frozen bits
                u[idx] = 0
            else:        # info bits
                if idx in qPC:
                    u[idx] = y0
                else:
                    u[idx] = inIntr[k] # Set information bits (interleaved)
                    k += 1
                    y0 = y0 ^ u[idx]
    else:
        # CRC-Aided Polar (CA-Polar)
        u[np.where(F == 0)] = inIntr   # Set information bits (interleaved)

    # Get G, nth Kronecker power of kernel
    n = int(np.log2(N))
    ak0 = np.array([[1, 0], [1, 1]])   # Arikan's kernel
    allG = [None] * n   # Initialize list
    allG[0] = ak0      # Assign first
    for i in range(1, n):
        allG[i] = np.kron(allG[i-1], ak0)
    G = allG[n-1]

    # Encode using matrix multiplication
    outd = np.mod(np.dot(u.T, G), 2).T
    out = outd.astype(inp.dtype)

    return out

def validateInputs(inp, E, nMax, iIL):
    # Check inputs

    fcnName = 'nrPolarEncode'

    # Validate single code-block input message
    if not isinstance(inp, np.ndarray):
        raise ValueError(f'{fcnName}: IN must be a numpy array')
    if inp.ndim == 2 and inp.shape[1] == 1:
        inp = inp.flatten()  # Convert (K,1) to 1D
    if inp.ndim != 1 or not np.all(np.logical_or(inp == 0, inp == 1)):
        raise ValueError(f'{fcnName}: IN must be a binary column vector')

    K = len(inp)

    # Validate base-2 logarithm of encoded message's maximum length (9 or 10)
    if not (isinstance(nMax, int) and nMax in [9, 10]):
        raise ValueError(f'{fcnName}: InvalidnMax')

    # Validate input interleaving flag
    if not isinstance(iIL, bool):
        raise ValueError(f'{fcnName}: IIL must be a boolean scalar')

    # A restriction for downlink (for up to 12 bits padding)
    # length K must be greater than or equal to 36 and less than or equal to 164
    if nMax == 9 and iIL and (K < 36 or K > 164):
        raise ValueError(f'nr5g:nrPolar:InvalidInputEncDLLength, {K}')

    # A restriction for uplink (for CA-Polar and PC-Polar)
    # length K must be greater than or equal to 18 and less than or equal to 1023,
    # with interim range from 26<=K<=30 not allowed
    if nMax == 10 and not iIL and (K < 18 or (K > 25 and K < 31) or K > 1023):
        raise ValueError(f'nr5g:nrPolar:InvalidInputEncULLength, {K}')
    if 18 <= K <= 25: # for PC-Polar
        nPC = 3
    else:
        nPC = 0

    # Validate rate-matched output length which must be less than or equal
    # to 8192 and greater than K+nPC
    if not (isinstance(E, int) and E > K + nPC and E <= 8192):
        raise ValueError(f'{fcnName}: E must be an integer > K+nPC and <= 8192')


