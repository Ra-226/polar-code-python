import numpy as np

def nrCRCEncode(blk, poly, mask=0):
    # nrCRCEncode Cyclic redundancy check calculation and appending
    #   BLKCRC = nrCRCEncode(BLK,POLY) calculates the CRC defined by POLY for
    #   input data block BLK and returns a copy of the input with the CRC
    #   appended in BLKCRC. BLK is a matrix (int or float) where
    #   each column is treated as a separate data block and processed
    #   independently. For the purpose of CRC calculation, any non-zero element
    #   of the input is treated as logical 1 while zeros are treated as logical
    #   0. The CRC polynomial is defined by a value from the set
    #   ('6','11','16','24A','24B','24C'). See TS 38.212 Section 5.1 for the
    #   associated polynomials.
    #
    #   BLKCRC = nrCRCEncode(BLK,POLY,MASK) behaves as above except the third
    #   parameter allows the appended CRC bits to be xor masked with the scalar
    #   nonnegative integer value of MASK. This mask is typically an RNTI. The
    #   MASK value is applied to the CRC bits MSB first/LSB last, i.e. (p0 xor
    #   m0),(p1 xor m1),...(pL-1 xor mL-1), where m0 is the MSB on the binary
    #   representation of the mask. If the mask value is greater than 2^L - 1,
    #   then LSB 'L' bits are considered for mask.
    #
    #   See also nrCRCDecode, nrCodeBlockSegmentLDPC, nrPolarEncode,
    #   nrLDPCEncode, nrRateMatchPolar, nrRateMatchLDPC, nrBCH, nrDCIEncode.
    #
    

    polyIndex = validateCRCinputs(blk, poly, mask, 'nrCRCEncode')

    polyLengths = [6, 11, 16, 24, 24, 24]
    gLen = polyLengths[polyIndex]

    # Perform cyclic redundancy check
    if blk.ndim == 1:
        blk = blk.reshape(-1, 1)
    codeLen, numCodeBlocks = blk.shape
    blkL = (blk != 0).astype(bool)  # Treat non-zero as True

    if codeLen == 0:
        return np.zeros((0, numCodeBlocks), dtype=blk.dtype)

    blkcrcL = np.zeros((codeLen + gLen, numCodeBlocks), dtype=bool)

    # In Python, implement the codegen path for CRC encoding
    gPoly = getPoly(poly)
    for i in range(numCodeBlocks):
        blkcrcL[:, i] = crcEncode(blkL[:, i].astype(float), gPoly, gLen)

    if mask != 0:
        # Convert decimal mask to bits (MSB first)
        maskBits = np.array([int(b) for b in np.binary_repr(mask, width=gLen)])
        maskBits = maskBits[::-1]  # Adjust to match MSB first
        blkcrcL[-gLen:, :] = np.logical_xor(blkcrcL[-gLen:, :], np.tile(maskBits[:, np.newaxis], (1, numCodeBlocks)))

    blkcrc = np.vstack((blk, blkcrcL[-gLen:, :].astype(blk.dtype)))

    return blkcrc

def crcEncode(inp, gPoly, gLen):
    # CRC Encode with all floats inputs, logical out.

    # Append zeros to the data block
    inPad = np.concatenate((inp, np.zeros(gLen)))

    # Perform cyclic redundancy check
    remBits = np.concatenate(([0], inPad[:gLen]))
    for i in range(len(inPad) - gLen):
        dividendBlk = np.concatenate((remBits[1:], [inPad[i + gLen]]))
        if dividendBlk[0] == 1:
            remBits = (gPoly + dividendBlk) % 2
        else:
            remBits = dividendBlk
    parityBits = remBits[1:]

    return np.concatenate((inp, parityBits)).astype(bool)

def getPoly(poly):
    # Initialize CRC polynomial (gPoly)

    poly = poly.upper()
    if poly == '6':
        return np.array([1, 1, 0, 0, 0, 0, 1])
    elif poly == '11':
        return np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    elif poly == '16':
        return np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    elif poly in ['24A', '24a']:
        return np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1])
    elif poly in ['24B', '24b']:
        return np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    else:  # '24C'
        return np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])

def nrCRCDecode(blkcrc, poly, mask=0):
    # nrCRCDecode Cyclic redundancy check decoding and removal
    #   [BLK,ERR] = nrCRCDecode(BLKCRC,POLY) returns BLK, the data only part of
    #   the combined data and CRC input BLKCRC, and uint32 ERR, the logical
    #   (xor) CRC difference. BLKCRC is a matrix (double, int8 or logical)
    #   where each column is treated as a separate data block and processed
    #   independently. The CRC polynomial is defined by a value from the set
    #   ('6','11','16','24A','24B','24C'). See TS 38.212 Section 5.1 for the
    #   associated polynomials.
    #
    #   [BLK,ERR] = nrCRCDecode(BLKCRC,POLY,MASK) behaves as above except the
    #   CRC difference is also XORed with the scalar nonnegative integer MASK
    #   parameter before it is returned in ERR.
    #
    #   See also nrCRCEncode, nrCodeBlockDesegmentLDPC, nrPolarDecode,
    #   nrLDPCDecode, nrRateRecoverPolar, nrRateRecoverLDPC, nrBCHDecode,
    #   nrDCIDecode.
    #
    

    polyIndex = validateCRCinputs(blkcrc, poly, mask, 'nrCRCDecode')

    polyLengths = [6, 11, 16, 24, 24, 24]
    gLen = polyLengths[polyIndex]

    if blkcrc.ndim == 1:
        blkcrc = blkcrc.reshape(-1, 1)
    codeLen, numCodeBlocks = blkcrc.shape

    if codeLen == 0:
        return np.zeros((0, numCodeBlocks), dtype=blkcrc.dtype), np.zeros((1, numCodeBlocks), dtype=np.uint32)

    # Perform cyclic redundancy check for data only part of blkcrc
    reEncodedBlk = nrCRCEncode(blkcrc[:-gLen, :], poly, mask)

    blk = reEncodedBlk[:-gLen, :]

    if codeLen <= gLen:
        # For input length less than parity bit length
        blkcrcL = np.vstack((np.zeros((gLen - codeLen, numCodeBlocks), dtype=bool), blkcrc > 0))
        if mask != 0:
            maskBits = np.array([int(b) for b in np.binary_repr(mask, width=gLen)])
            maskBits = maskBits[::-1]
            errBits = np.logical_xor(blkcrcL, np.tile(maskBits[:, np.newaxis], (1, numCodeBlocks)))
        else:
            errBits = blkcrcL
    else:
        errBits = np.logical_xor(reEncodedBlk[-gLen:, :] > 0, blkcrc[-gLen:, :] > 0)

    # Compute err as uint32
    powers = 2 ** np.arange(gLen - 1, -1, -1)
    err = np.sum(errBits.astype(np.uint32) * np.tile(powers[:, np.newaxis], (1, numCodeBlocks)), axis=0).astype(np.uint32)

    return blk, err

def validateCRCinputs(inp, poly, mask, fcnName):
    # validateCRCinputs validates the inputs of CRC encoding and decoding
    #
    #    polyIndex = validateCRCinputs(in,poly,mask,fcnName)
    #    validates the inputs as specified below and returns an index associated
    #    with the polynomial:
    #    1) Input data block (in) should be a matrix (int8, double or
    #    logical)
    #    2) CRC polynomial (poly) should be a character row vector or a scalar
    #    string from one of the following ('6','11','16','24A','24B','24C')
    #    3) Mask should be a scalar nonnegative integer
    #    4) Function name (fcnName) should be the name of calling function
    #

    if fcnName.lower() == 'nrcrcencode':
        input_name = 'BLK'
    else:
        input_name = 'BLKCRC'

    # Validate inp: 2D, real, non-nan
    if not isinstance(inp, np.ndarray) or inp.ndim > 2 or np.any(np.iscomplex(inp)) or np.any(np.isnan(inp)):
        raise ValueError(f"{fcnName}: {input_name} must be a 2D real non-NaN array")

    # Validate poly
    polyList = ['6', '11', '16', '24A', '24B', '24C']
    poly = poly.upper()
    if poly not in polyList:
        raise ValueError(f"{fcnName}: POLY must be one of {polyList}")

    polyIndex = polyList.index(poly)

    # Validate mask: scalar, integer, nonnegative
    if not (isinstance(mask, (int, float)) and mask >= 0 and mask == int(mask)):
        raise ValueError(f"{fcnName}: MASK must be a scalar nonnegative integer")

    return polyIndex
