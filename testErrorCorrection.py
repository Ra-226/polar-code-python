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

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Channel BER vs Performance
ax1.plot(ber_list, results_ber, 'bo-', label='Decoded BER', linewidth=2, markersize=8)
ax1.plot(ber_list, results_fer, 'rs-', label='FER', linewidth=2, markersize=8)
ax1.plot(ber_list, ber_list, 'k--', label='Channel BER', linewidth=1)
ax1.set_xlabel('Channel BER')
ax1.set_ylabel('Error Rate')
ax1.set_title('Error Correction vs Channel BER (L=8)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, max(max(results_ber), max(results_fer)) * 1.1 + 0.01])

# Plot 2: List Length vs Performance
ax2.plot(L_list, results_L_ber, 'go-', label='Decoded BER', linewidth=2, markersize=8)
ax2.plot(L_list, results_L_fer, 'ms-', label='FER', linewidth=2, markersize=8)
ax2.set_xlabel('List Length L')
ax2.set_ylabel('Error Rate')
ax2.set_title('List Length Impact (Channel BER=0.20)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, max(max(results_L_ber), max(results_L_fer)) * 1.1 + 0.01])

plt.tight_layout()
plt.savefig('polar_test_results.png', dpi=150, bbox_inches='tight')
print('\nPlot saved: polar_test_results.png')
plt.show()
