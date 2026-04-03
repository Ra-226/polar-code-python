import numpy as np

from utils import *
from nrCRC import nrCRCDecode


def nrPolarDecode(inp, K, E, L, *varargin):
    # nrPolarDecode Polar decode
    #   DECBITS = nrPolarDecode(REC,K,E,L) decodes the rate-recovered input,
    #   REC, for a (N,K) Polar code, using a CRC-aided successive-cancellation
    #   list decoder, with the list length specified by L. The input REC is a
    #   column vector of length N (a power of 2), representing the
    #   log-likelihood ratios as soft inputs to the decoder. K is the number of
    #   message bits, and E is the rate-matched output length. The output
    #   DECBITS is a column vector of length K.
    #
    #   DECBITS = nrPolarDecode(...,PADCRC) specifies whether the input was
    #   prepadded by ones prior to the CRC encoding with all-zeros register
    #   state on the transmit end. PADCRC must be a boolean scalar where for a
    #   true value, the input is assumed to be prepadded with ones, while a
    #   false value indicates no prepadding was used. The default is false.
    #
    #   DECBITS = nrPolarDecode(...,PADCRC,RNTI) also specifies the RNTI value
    #   that may have been used at the transmit end for masking. The default is
    #   0.
    #
    #   DECBITS = nrPolarDecode(REC,K,E,L,NMAX,IIL,CRCLEN) specifies the three
    #   parameter set of: NMAX (an integer value of either 9 or 10), IIL (a
    #   boolean scalar) and CRCLEN (an integer value one of 24, 11 or 6). The
    #   allowed value sets of {9,true,24} and {10,false,11},{10,false,6} for
    #   {NMAX,IIL,CRCLEN} apply for downlink and uplink configurations. When
    #   the three parameters are not specified, the value set for the downlink
    #   configuration is used. PADCRC is assumed false and RNTI to be 0 for
    #   this syntax.
    #
    #   See also nrPolarEncode, nrRateRecoverPolar, nrCRCDecode, nrDCIDecode,
    #   nrBCHDecode, nrUCIDecode.
    #

    nargin = 4 + len(varargin)

    if nargin == 4:
        # Downlink parameters
        # nrPolarDecode(inp,K,E,L)
        nMax = 9
        iIL = True
        crcLen = 24

        padCRC = False           # default, for BCH and UCI
        rnti = 0                 # default, unused

    elif nargin == 5:
        # Downlink parameters for which padCRC applies
        # nrPolarDecode(inp,K,E,L,padCRC)
        nMax = 9
        iIL = True
        crcLen = 24

        padCRC = varargin[0]     # true for DCI
        rnti = 0                 # default, unused

    elif nargin == 6:
        # Downlink parameters for which padCRC, RNTI apply
        # nrPolarDecode(inp,K,E,L,padCRC,RNTI)
        nMax = 9
        iIL = True
        crcLen = 24
        padCRC = varargin[0]     # true for DCI
        rnti = varargin[1]       # user-specified, only for DCI

    elif nargin == 7:
        # Support both downlink and uplink
        # nrPolarDecode(inp,K,E,L,nMax,iIL,crcLen)
        nMax = varargin[0]
        iIL = varargin[1]
        crcLen = varargin[2]

        padCRC = False           # default, for BCH and UCI
        rnti = 0                 # default, unused

    # Input is a single code block and assumes CRC bits are included in K
    # output
    validateInputs(inp, K, E, L, nMax, iIL, crcLen, padCRC, rnti)

    # F accounts for nPC bits, if present
    F, qPC, _ = construct(K, E, nMax)

    # CA-SCL decode
    outkpc = lclPolarDecode(inp, F, L, iIL, crcLen, padCRC, rnti, qPC)

    # Remove nPC bits from output, if present
    if len(qPC) > 0:
        # Extract the information only bits
        qI = np.where(F == 0)[0]
        k = 0
        out = np.zeros(len(outkpc) - 3, dtype=int)
        for idx in range(len(qI)):
            if qI[idx] not in qPC:
                out[k] = outkpc[idx]
                k += 1
    else:
        out = outkpc

    return out

def validateInputs(inp, K, E, L, nMax, iIL, crcLen, padCRC, rnti):
    # Check inputs

    fcnName = 'nrPolarDecode'

    # Validate rate-recovered input for a single code-block
    if not (isinstance(inp, np.ndarray) and inp.dtype in [np.float32, np.float64] and inp.ndim == 1):
        raise ValueError(f"{fcnName}: REC must be a real column vector")

    N = len(inp)
    if int(np.log2(N)) != np.log2(N):
        raise ValueError('nr5g:nrPolar:InvalidInputDecLength')

    # Validate the number of message bits which must be less than N
    if not (isinstance(K, int) and K < N):
        raise ValueError(f"{fcnName}: K must be an integer < N")

    # Validate base-2 logarithm of rate-recovered input's maximum length (9 or 10)
    if not (isinstance(nMax, int) and nMax in [9, 10]):
        raise ValueError(f"{fcnName}: InvalidnMax")

    # Validate output deinterleaving flag
    if not isinstance(iIL, bool):
        raise ValueError(f"{fcnName}: IIL must be a boolean scalar")

    # Validate the number of appended CRC bits (6, 11 or 24)
    if not (isinstance(crcLen, int) and crcLen in [6, 11, 24]):
        raise ValueError(f"{fcnName}: InvalidCRCLen")

    # Validate CRC prepadding flag
    if not isinstance(padCRC, bool):
        raise ValueError(f"{fcnName}: PADCRC must be boolean")

    # Validate RNTI
    if not (isinstance(rnti, int) and 0 <= rnti <= 65535):
        raise ValueError(f"{fcnName}: RNTI must be integer between 0 and 65535")

    # A restriction for downlink (for up to 12 bits padding)
    # length K must be greater than or equal to 36 and less than or equal
    # to 164
    if nMax == 9 and iIL and crcLen == 24 and (K < 36 or K > 164):
        raise ValueError('nr5g:nrPolar:InvalidKDL')

    # A restriction for uplink (for CA-Polar)
    # length K must be greater than 30 and less than or equal to 1023
    if nMax == 10 and not iIL and crcLen == 11 and not padCRC and (K <= 30 or K > 1023):
        raise ValueError('nr5g:nrPolar:InvalidKUL')

    # A restriction for uplink (for PC-Polar)
    # length K must be greater than 17 and less than or equal to 25
    if nMax == 10 and not iIL and crcLen == 6 and not padCRC and (K < 18 or K > 25):
        raise ValueError('nr5g:nrPolar:InvalidKULPC')

    # Validate rate-matched output length which must be less than or equal
    # to 8192 and greater than K
    if not (isinstance(E, int) and K < E <= 8192):
        raise ValueError(f"{fcnName}: E must be an integer > K and <= 8192")

    # Validate decoding list length (assuming nr5g.internal.validateParameters is missing, placeholder check)
    if not (isinstance(L, int) and L > 0):
        raise ValueError(f"{fcnName}: ListLength must be positive integer")

def lclPolarDecode(inp, F, L, iIL, crcLen, padCRC, rnti, qPC):
    # References:
    # [1] Tal, I, and Vardy, A., "List decoding of Polar Codes", IEEE
    # Transactions on Information Theory, vol. 61, No. 5, pp. 2213-2226,
    # May 2015.
    # [2] Stimming, A. B., Parizi, M. B., and Burg, A., "LLR-Based
    # Successive Cancellation List Decoding of Polar Codes", IEEE
    # Transaction on Signal Processing, vol. 63, No. 19, pp.5165-5179,
    # 2015.

    # Setup
    N = len(F)
    m = int(np.log2(N))
    K = np.sum(F == 0)  # includes nPC bits as well

    # CRCs as per TS 38.212, Section 5.1
    if crcLen == 24:         # '24C', downlink
        polyStr = '24C'
    elif crcLen == 11:     # '11', uplink
        polyStr = '11'
    else:  # crcLen == 6      # '6', uplink
        polyStr = '6'

    br = np.zeros(N, dtype=int)
    for idxBit in range(N):
        br[idxBit] = polarBitReverse(idxBit, m)

    if iIL:
        piInterl = interleaveMap(K)
    else:
        piInterl = np.arange(K)

    # Initialize core
    sttStr, arrayPtrLLR, arrayPtrC = initializeDataStructures(N, L, m)
    iniPathIdx, sttStr = assignInitialPath(sttStr)
    sp, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrP(sttStr, arrayPtrLLR, arrayPtrC, 1, iniPathIdx)
    arrayPtrLLR[1][sp][:] = inp[br]

    mplus1 = m + 1

    # Main loop
    for phase in range(1, N + 1):
        sttStr, arrayPtrLLR, arrayPtrC = recursivelyCalcP(sttStr, arrayPtrLLR, arrayPtrC, mplus1, phase)
        
        pm2 = (phase - 1) % 2
        if F[phase - 1] == 1:  # 0-based index
            # Path for frozen (and punctured) bits
            for pathIdx in range(L):
                if not sttStr['activePath'][pathIdx]:
                    continue
                sc, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrC(sttStr, arrayPtrLLR, arrayPtrC, mplus1, pathIdx)
                arrayPtrC[mplus1][sc][0, pm2] = 0  # set to 0

                # Revised approximation metric update
                tmp = arrayPtrLLR[mplus1][sc][0]
                if tmp < 0:
                    sttStr['llrPathMetric'][pathIdx] += abs(tmp)
        else:  # Path for info bits
            sttStr, arrayPtrLLR, arrayPtrC = contPathsUnfrozenBit(sttStr, arrayPtrLLR, arrayPtrC, phase)

        if pm2 == 1:
            # For pm2 == 1:
            sttStr, arrayPtrLLR, arrayPtrC = recursivelyUpdateC(sttStr, arrayPtrLLR, arrayPtrC, mplus1, phase)

    # Return the best codeword in the list. Use CRC checks, if enabled
    pathIdx1 = 1
    p1 = np.finfo(float).max
    crcCW = False
    for pathIdx in range(L):
        if not sttStr['activePath'][pathIdx]:
            continue

        # For best path:
        sc, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrC(sttStr, arrayPtrLLR, arrayPtrC, mplus1, pathIdx)
        if crcLen > 0:
            canCW = sttStr['savedCWs'][:, sc - 1]  # N, with frozen bits
            canMsg = canCW[F == 0]         # K bits only (with nPC)
            canMsg[piInterl] = canMsg      # deinterleave (for k+nPC)

            if len(qPC) > 0:
                # Extract the info only bits, minus the PC ones
                qI = np.where(F == 0)[0]
                k = 0
                out = np.zeros(len(canMsg) - len(qPC), dtype=int)
                for idx in range(len(qI)):
                    if qI[idx] not in qPC:
                        out[k] = canMsg[idx]
                        k += 1
            else:
                out = canMsg

            # Check CRC: errFlag is 1 for error, 0 for no errors
            if padCRC:  # prepad with ones
                padCRCMsg = np.concatenate((np.ones(crcLen), out))
            else:
                padCRCMsg = out
            _, errFlag = nrCRCDecode(padCRCMsg, polyStr, rnti)
            if errFlag:      # !=0 => fail
                continue     # move to next path

        crcCW = True
        if p1 > sttStr['llrPathMetric'][pathIdx]:
            p1 = sttStr['llrPathMetric'][pathIdx]
            pathIdx1 = pathIdx

    if not crcCW:   # no codeword found which passes crcCheck
        pathIdx1 = 1
        p1 = np.finfo(float).max
        for pathIdx in range(L):
            if not sttStr['activePath'][pathIdx]:
                continue

            if p1 > sttStr['llrPathMetric'][pathIdx]:
                p1 = sttStr['llrPathMetric'][pathIdx]
                pathIdx1 = pathIdx

    # Get decoded bits
    sc, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrC(sttStr, arrayPtrLLR, arrayPtrC, mplus1, pathIdx1)
    decCW = sttStr['savedCWs'][:, sc - 1]      # N, with frozen bits
    dec = decCW[F == 0]                        # K, info + nPC bits only
    dec[piInterl] = dec                        # Deinterleave output, K+nPC

    return dec

def contPathsUnfrozenBit(sttStr, arrayPtrLLR, arrayPtrC, phase):
    # Input:
    #   phase: phase phi, 1-based, 1:2^m, or 1:N
    #
    # Revised metric update to use approximation:
    #   log(1+exp(x)) = 0 for x < 0,
    #                 = x for x >= 0.

    # Populate probForks
    probForks = np.full((sttStr['L'], 2), -np.finfo(float).max)
    i = 0
    mplus1 = sttStr['m'] + 1
    for pathIdx in range(sttStr['L']):
        if sttStr['activePath'][pathIdx]:
            sp, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrP(sttStr, arrayPtrLLR, arrayPtrC, mplus1, pathIdx)

            # Revised approximation use
            tmp = arrayPtrLLR[mplus1][sp][0]
            if tmp > 0:
                probForks[pathIdx, 0] = - sttStr['llrPathMetric'][pathIdx]
                probForks[pathIdx, 1] = - (sttStr['llrPathMetric'][pathIdx] + tmp)
            else:
                probForks[pathIdx, 0] = - (sttStr['llrPathMetric'][pathIdx] + abs(tmp))
                probForks[pathIdx, 1] = - sttStr['llrPathMetric'][pathIdx]
            i += 1

    rho = min(2 * i, sttStr['L'])
    contForks = np.zeros((sttStr['L'], 2), dtype=bool)
    # Populate contForks such that contForks(l,b) is true iff
    # probForks(l,b) is one of rho largest entries in probForks.
    prob = np.sort(probForks.flatten())[::-1]
    if rho > 0:
        threshold = prob[rho - 1]
    else:
        threshold = prob[0]  # Largest

    numPop = 0
    for pathIdx in range(sttStr['L']):
        for bIdx in range(2):
            if numPop == rho:
                break
            if probForks[pathIdx, bIdx] > threshold:
                contForks[pathIdx, bIdx] = True
                numPop += 1

    if numPop < rho:
        for pathIdx in range(sttStr['L']):
            for bIdx in range(2):
                if numPop == rho:
                    break
                if probForks[pathIdx, bIdx] == threshold:
                    contForks[pathIdx, bIdx] = True
                    numPop += 1

    # First, kill-off non-continuing paths
    for pathIdx in range(sttStr['L']):
        if sttStr['activePath'][pathIdx] and not contForks[pathIdx, 0] and not contForks[pathIdx, 1]:
            sttStr = killPath(sttStr, pathIdx)

    # Continue relevant paths, duplicating if necessary
    pm2 = (phase - 1) % 2
    for pathIdx in range(sttStr['L']):
        if not contForks[pathIdx, 0] and not contForks[pathIdx, 1]:
            # Both forks are bad
            continue

        sc, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrC(sttStr, arrayPtrLLR, arrayPtrC, mplus1, pathIdx)
        if contForks[pathIdx, 0] and contForks[pathIdx, 1]:
            # Both forks are good
            arrayPtrC[mplus1][sc][0, pm2] = 0
            sttStr['savedCWs'][phase - 1, sc] = 0

            pathIdx1, sttStr = clonePath(sttStr, pathIdx)
            sc2, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrC(sttStr, arrayPtrLLR, arrayPtrC, mplus1, pathIdx1)
            sttStr['savedCWs'][:phase - 1, sc2] = sttStr['savedCWs'][:phase - 1, sc]

            arrayPtrC[mplus1][sc2][0, pm2] = 1
            sttStr['savedCWs'][phase - 1, sc2] = 1
            # Revised approximation metric update for 0
            tmp = arrayPtrLLR[mplus1][sc][0]
            if tmp < 0:
                sttStr['llrPathMetric'][pathIdx] += abs(tmp)
            # For 1
            tmp2 = arrayPtrLLR[mplus1][sc2][0]
            if tmp2 > 0:
                sttStr['llrPathMetric'][pathIdx1] += tmp2
        else:
            # Exactly one fork is good
            tmp = arrayPtrLLR[mplus1][sc][0]
            if contForks[pathIdx, 0]:
                arrayPtrC[mplus1][sc][0, pm2] = 0
                sttStr['savedCWs'][phase - 1, sc] = 0
                if tmp < 0:
                    sttStr['llrPathMetric'][pathIdx] += abs(tmp)
            else:
                arrayPtrC[mplus1][sc][0, pm2] = 1
                sttStr['savedCWs'][phase - 1, sc] = 1
                if tmp > 0:
                    sttStr['llrPathMetric'][pathIdx] += tmp

    return sttStr, arrayPtrLLR, arrayPtrC

def recursivelyCalcP(sttStr, arrayPtrLLR, arrayPtrC, layer, phase):
    # Input:
    #   layer: layer lambda, 1-based, 1:m+1
    #   phase: phase phi, 1-based, 1:2^layer or 1:N

    if layer == 1:
        return sttStr, arrayPtrLLR, arrayPtrC

    psi = ((phase - 1) // 2) + 1
    pm2 = (phase - 1) % 2
    if pm2 == 0:
        sttStr, arrayPtrLLR, arrayPtrC = recursivelyCalcP(sttStr, arrayPtrLLR, arrayPtrC, layer - 1, psi)

    expm = 2 ** (sttStr['m'] + 1 - layer)
    for pathIdx in range(sttStr['L']):
        if not sttStr['activePath'][pathIdx]:
            continue

        sp, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrP(sttStr, arrayPtrLLR, arrayPtrC, layer, pathIdx)
        spminus1, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrP(sttStr, arrayPtrLLR, arrayPtrC, layer - 1, pathIdx)
        sc, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrC(sttStr, arrayPtrLLR, arrayPtrC, layer, pathIdx)
        for beta in range(expm):
            aa = arrayPtrLLR[layer - 1][spminus1][2 * beta]
            bb = arrayPtrLLR[layer - 1][spminus1][2 * beta + 1]
            if pm2 == 0:
                arrayPtrLLR[layer][sp][beta] = np.sign(aa) * np.sign(bb) * min(abs(aa), abs(bb))
            else:
                u1 = arrayPtrC[layer][sc][beta, 0]
                arrayPtrLLR[layer][sp][beta] = ((-1)**u1 * aa + bb)

    return sttStr, arrayPtrLLR, arrayPtrC

def recursivelyUpdateC(sttStr, arrayPtrLLR, arrayPtrC, layer, phase):
    # Input:
    #   layer: layer lambda, 1-based, 1:m+1
    #   phase: phase phi, 1-based, 1:2^layer or 1:N, must be odd

    psi = (phase - 1) // 2
    pm2 = psi % 2
    expm = 2 ** (sttStr['m'] + 1 - layer)
    for pathIdx in range(sttStr['L']):
        if not sttStr['activePath'][pathIdx]:
            continue
        sc, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrC(sttStr, arrayPtrLLR, arrayPtrC, layer, pathIdx)
        scminus1, sttStr, arrayPtrLLR, arrayPtrC = getArrayPtrC(sttStr, arrayPtrLLR, arrayPtrC, layer - 1, pathIdx)
        for beta in range(expm):
            arrayPtrC[layer - 1][scminus1][2 * beta, pm2] = np.logical_xor(arrayPtrC[layer][sc][beta, 0], arrayPtrC[layer][sc][beta, 1]).astype(int)
            arrayPtrC[layer - 1][scminus1][2 * beta + 1, pm2] = arrayPtrC[layer][sc][beta, 1]

    if pm2 == 1:
        sttStr, arrayPtrLLR, arrayPtrC = recursivelyUpdateC(sttStr, arrayPtrLLR, arrayPtrC, layer - 1, psi + 1)

    return sttStr, arrayPtrLLR, arrayPtrC

def initializeDataStructures(N, L, m):
    # Indices are 1-based for layers to match MATLAB

    sttStr = {}
    sttStr['L'] = L
    sttStr['m'] = m                   # log2(N)

    mplus1 = m + 1
    arrayPtrLLR = [None] * (mplus1 + 1)
    arrayPtrC = [None] * (mplus1 + 1)
    for layer in range(1, mplus1 + 1):
        expm = 2 ** (m + 1 - layer)
        arrayPtrLLR[layer] = [np.zeros(expm) for _ in range(L)]
        arrayPtrC[layer] = [np.zeros((expm, 2), dtype=int) for _ in range(L)]

    sttStr['llrPathMetric'] = np.zeros(L)

    sttStr['pathIdxToArrayIdx'] = np.full((mplus1 + 1, L), -1, dtype=int)   # (m+2)-by-L, -1 for invalid, 1-based layers

    sttStr['inactiveArrayIndices'] = [None] * (mplus1 + 1)
    sttStr['inactiveArrayIndicesLen'] = np.zeros(mplus1 + 1, dtype=int)
    for layer in range(1, mplus1 + 1):
        sttStr['inactiveArrayIndices'][layer] = np.arange(L)
        sttStr['inactiveArrayIndicesLen'][layer] = L

    sttStr['arrayReferenceCount'] = np.zeros((mplus1 + 1, L), dtype=int)

    sttStr['inactivePathIndices'] = np.arange(L)     # 0-based
    sttStr['inactivePathIndicesLen'] = L        # 1-by-1, stack depth
    sttStr['activePath'] = np.zeros(L, dtype=bool) # no paths are active

    sttStr['savedCWs'] = np.zeros((N, L), dtype=int)             # saved codewords

    return sttStr, arrayPtrLLR, arrayPtrC

def assignInitialPath(sttStr):
    # Output:
    #   pathIdx: initial path index l, 1-based, 1:L

    pathIdx = sttStr['inactivePathIndices'][sttStr['inactivePathIndicesLen'] - 1]  # 0-based
    sttStr['inactivePathIndicesLen'] -= 1
    sttStr['activePath'][pathIdx] = True

    # Associate arrays with path index
    for layer in range(1, sttStr['m'] + 2):
        s = sttStr['inactiveArrayIndices'][layer][sttStr['inactiveArrayIndicesLen'][layer] - 1]  # 0-based
        sttStr['inactiveArrayIndicesLen'][layer] -= 1

        sttStr['pathIdxToArrayIdx'][layer, pathIdx] = s
        sttStr['arrayReferenceCount'][layer, s] = 1

    return pathIdx, sttStr

def clonePath(sttStr, pathIdx):
    # Input:
    #   pathIdx: path index to clone, l, 1-based, 1:L
    # Output:
    #   clPathIdx: cloned path index, l', 1-based

    clPathIdx = sttStr['inactivePathIndices'][sttStr['inactivePathIndicesLen'] - 1]  # 0-based
    sttStr['inactivePathIndicesLen'] -= 1
    sttStr['activePath'][clPathIdx] = True
    sttStr['llrPathMetric'][clPathIdx] = sttStr['llrPathMetric'][pathIdx]

    # Make clPathIdx reference same arrays as pathIdx
    for layer in range(1, sttStr['m'] + 2):
        s = sttStr['pathIdxToArrayIdx'][layer, pathIdx]
        sttStr['pathIdxToArrayIdx'][layer, clPathIdx] = s
        sttStr['arrayReferenceCount'][layer, s] += 1

    return clPathIdx, sttStr

def killPath(sttStr, pathIdx):
    # Input:
    #   pathIdx: path index to kill, l, 1-based, 1:L

    # Mark path pathIdx as inactive
    sttStr['activePath'][pathIdx] = False
    sttStr['inactivePathIndicesLen'] += 1
    sttStr['inactivePathIndices'][sttStr['inactivePathIndicesLen'] - 1] = pathIdx
    sttStr['llrPathMetric'][pathIdx] = 0

    # Disassociate arrays with path Idx
    for layer in range(1, sttStr['m'] + 2):
        s = sttStr['pathIdxToArrayIdx'][layer, pathIdx]
        sttStr['arrayReferenceCount'][layer, s] -= 1

        if sttStr['arrayReferenceCount'][layer, s] == 0:
            sttStr['inactiveArrayIndicesLen'][layer] += 1
            sttStr['inactiveArrayIndices'][layer][sttStr['inactiveArrayIndicesLen'][layer] - 1] = s

    return sttStr

def getArrayPtrP(sttStr, arrayPtrLLR, arrayPtrC, layer, pathIdx):
    # Input:
    #   layer:   layer lambda, 1-based, 1:m+1
    #   pathIdx: path index l, 1-based, 1:L
    # Output:
    #   s2: corresponding pathIdx for same layer

    s = sttStr['pathIdxToArrayIdx'][layer, pathIdx]  # 0-based
    if s == -1:
        raise IndexError("Invalid array index for layer {} path {}".format(layer, pathIdx))
    if sttStr['arrayReferenceCount'][layer, s] == 1:
        s2 = s
    else:
        s2 = sttStr['inactiveArrayIndices'][layer][sttStr['inactiveArrayIndicesLen'][layer] - 1]  # 0-based
        sttStr['inactiveArrayIndicesLen'][layer] -= 1

        # deep copy
        arrayPtrLLR[layer][s2] = arrayPtrLLR[layer][s].copy()
        arrayPtrC[layer][s2] = arrayPtrC[layer][s].copy()

        sttStr['arrayReferenceCount'][layer, s] -= 1
        sttStr['arrayReferenceCount'][layer, s2] = 1
        sttStr['pathIdxToArrayIdx'][layer, pathIdx] = s2

    return s2, sttStr, arrayPtrLLR, arrayPtrC

def getArrayPtrC(sttStr, arrayPtrLLR, arrayPtrC, layer, pathIdx):
    # Input:
    #   layer:   layer lambda, 1-based, 1:m+1
    #   pathIdx: path index l, 1-based, 1:L
    # Output:
    #   s2: corresponding pathIdx for same layer

    s = sttStr['pathIdxToArrayIdx'][layer, pathIdx]  # 0-based
    if s == -1:
        raise IndexError("Invalid array index for layer {} path {}".format(layer, pathIdx))
    if sttStr['arrayReferenceCount'][layer, s] == 1:
        s2 = s
    else:
        s2 = sttStr['inactiveArrayIndices'][layer][sttStr['inactiveArrayIndicesLen'][layer] - 1]  # 0-based
        sttStr['inactiveArrayIndicesLen'][layer] -= 1

        # deep copy
        arrayPtrC[layer][s2] = arrayPtrC[layer][s].copy()
        arrayPtrLLR[layer][s2] = arrayPtrLLR[layer][s].copy()

        sttStr['arrayReferenceCount'][layer, s] -= 1
        sttStr['arrayReferenceCount'][layer, s2] = 1
        sttStr['pathIdxToArrayIdx'][layer, pathIdx] = s2

    return s2, sttStr, arrayPtrLLR, arrayPtrC

def polarBitReverse(b, n):
    # polarBitReverse Bit-wise reverse input value
    #
    #   BR = polarBitReverse(B,N) returns the bit-wise reversed-value of B,
    #   each represented over N bits.
    #
    

    # Convert to binary string, reverse, and convert back
    bin_str = np.binary_repr(b, width=n)
    rev_str = bin_str[::-1]
    return int(rev_str, 2)
