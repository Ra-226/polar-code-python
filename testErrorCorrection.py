import numpy as np
import matplotlib.pyplot as plt
from nrCRC import nrCRCEncode, nrCRCDecode
from nrPolarEncode import nrPolarEncode
from nrPolarDecode import nrPolarDecode

plt.rcParams['font.family'] = 'sans-serif'

K_info = 64
crcLen = 11
K_total = K_info + crcLen
E = 512
nMax = 9
iIL = True
crc_poly = '11'

print('Polar Error Correction Test')
print('K=%d, CRC=%d, E=%d, Rate=%.3f\n' % (K_info, crcLen, E, K_info/E))

# Test 1: Different channel BER
print('=== Test 1: Channel BER (L=8, 20 frames) ===')
ber_list = [0.10, 0.15, 0.20, 0.25, 0.30]
L = 8
numFrames = 20

results_ber = []
results_fer = []

for ber_channel in ber_list:
    bitErrs = 0
    frmErrs = 0

    for frm in range(numFrames):
        infoBits = np.random.randint(0, 2, K_info)
        msgCRC = nrCRCEncode(infoBits, crc_poly).flatten()
        encBits = nrPolarEncode(msgCRC, E, nMax, iIL).astype(np.float64)

        flips = np.random.rand(len(encBits)) < ber_channel
        rxBits = np.logical_xor(encBits, flips).astype(np.float64)

        llr_scale = np.log((1 - ber_channel) / ber_channel)
        llr = llr_scale * (1 - 2 * rxBits)

        decBits = nrPolarDecode(llr, K_total, E, L, nMax, iIL, crcLen)
        rxInfo, crcErr = nrCRCDecode(decBits, crc_poly)
        rxInfo = rxInfo.flatten()

        bitErrs += np.sum(rxInfo != infoBits)
        if np.sum(rxInfo != infoBits) > 0 or crcErr != 0:
            frmErrs += 1

    ber = bitErrs / (numFrames * K_info)
    fer = frmErrs / numFrames
    results_ber.append(ber)
    results_fer.append(fer)

    print('Channel BER=%.2f: Decoded BER=%.4f, FER=%.4f' % (ber_channel, ber, fer))

# Test 2: Different list length L
print('\n=== Test 2: List Length L (Channel BER=0.20, 20 frames) ===')
L_list = [1, 2, 4, 8, 16]
ber_channel = 0.20

results_L_ber = []
results_L_fer = []

for L in L_list:
    bitErrs = 0
    frmErrs = 0

    for frm in range(numFrames):
        infoBits = np.random.randint(0, 2, K_info)
        msgCRC = nrCRCEncode(infoBits, crc_poly).flatten()
        encBits = nrPolarEncode(msgCRC, E, nMax, iIL).astype(np.float64)

        flips = np.random.rand(len(encBits)) < ber_channel
        rxBits = np.logical_xor(encBits, flips).astype(np.float64)

        llr_scale = np.log((1 - ber_channel) / ber_channel)
        llr = llr_scale * (1 - 2 * rxBits)

        decBits = nrPolarDecode(llr, K_total, E, L, nMax, iIL, crcLen)
        rxInfo, crcErr = nrCRCDecode(decBits, crc_poly)
        rxInfo = rxInfo.flatten()

        bitErrs += np.sum(rxInfo != infoBits)
        if np.sum(rxInfo != infoBits) > 0 or crcErr != 0:
            frmErrs += 1

    ber = bitErrs / (numFrames * K_info)
    fer = frmErrs / numFrames
    results_L_ber.append(ber)
    results_L_fer.append(fer)

    print('L=%2d: Decoded BER=%.4f, FER=%.4f' % (L, ber, fer))
