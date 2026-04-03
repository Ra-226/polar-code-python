import numpy as np

from nrCRC import nrCRCEncode, nrCRCDecode
from nrPolarEncode import nrPolarEncode
from nrPolarDecode import nrPolarDecode

# ========================================================================
#  The following are local functions used in this script
#  ========================================================================

def bscChannelLLR(encodedBits, pFlip, llrMag):
    # Binary Symmetric Channel (BSC) LLR generation: First flip by pFlip, then use fixed magnitude LLR
    flips = np.random.rand(len(encodedBits)) < pFlip
    rxBits = np.logical_xor(encodedBits, flips).astype(np.float64)
    llr = (1 - 2 * rxBits) * llrMag  # 0 -> +llrMag, 1 -> -llrMag
    return llr

def awgnChannelLLR(encodedBits, EbN0dB, codeRate):
    # LLR calculation on BPSK + AWGN channel
    # encodedBits: 0/1, bits
    # codeRate: information bits / encoded bits (for Eb/N0 conversion)
    x = 1 - 2 * encodedBits.astype(float)  # BPSK: 0->+1, 1->-1
    
    EbN0 = 10 ** (EbN0dB / 10)
    # For BPSK, Es = Eb; noise one-sided power spectral density N0 = 1/EbN0
    N0 = 1 / EbN0
    sigma2 = N0 / 2
    
    noise = np.sqrt(sigma2) * np.random.randn(len(x))
    y = x + noise
    
    # LLR = 2*y / sigma^2 for BPSK in AWGN
    llr = 2 * y / sigma2
    return llr

def burstChannelLLR(encodedBits, pFlip, burstLen, llrMag):
    # Burst error channel: Try to concentrate flips into consecutive segments
    N = len(encodedBits)
    rxBits = encodedBits.copy()
    
    # Estimate how many bursts are needed to approximately achieve overall pFlip
    numFlipsTarget = round(pFlip * N)
    numBursts = max(1, round(numFlipsTarget / burstLen))
    
    for b in range(numBursts):
        if N <= burstLen:
            startIdx = 0
        else:
            startIdx = np.random.randint(0, N - burstLen + 1)
        idx = np.arange(startIdx, min(startIdx + burstLen, N))
        rxBits[idx] = 1 - rxBits[idx]  # Flip the entire segment
    
    rxBits = rxBits.astype(np.float64)
    llr = (1 - 2 * rxBits) * llrMag
    return llr

def erasureChannelLLR(encodedBits, erasureFraction, baseLLR):
    # Erasure channel: First give a base LLR, then randomly set some bits' LLR to 0 (completely uncertain)
    N = len(encodedBits)
    rxBits = encodedBits
    rxBits = rxBits.astype(np.float64)
    llr = (1 - 2 * rxBits) * baseLLR
    
    numErase = round(erasureFraction * N)
    if numErase > 0:
        eraseIdx = np.random.choice(N, numErase, replace=False)
        llr[eraseIdx] = 0  # 0 means "no information"
    
    return llr

def mp3LikeChannel(encodedBits, codeRate):
    # Rough simulation of "MP3 compression + noise + resampling" comprehensive effect channel
    # This is just a toy model for testing Polar + interleaving robustness
    N = len(encodedBits)
    
    # 1) Base AWGN
    EbN0dB = 4  # Slightly lower SNR
    llr = awgnChannelLLR(encodedBits, EbN0dB, codeRate)
    
    # 2) Simulate some frequency bands completely compressed or thresholded: erase a portion
    numErase = round(0.1 * N)  # 10% erasure
    if numErase > 0:
        eraseIdx = np.random.choice(N, numErase, replace=False)
        llr[eraseIdx] = 0
    
    # 3) Simulate distortion introduced by resampling / compression: reverse symbols in some consecutive segments
    numBursts = 5
    burstLen = round(0.03 * N)  # 3% length each time
    for b in range(numBursts):
        if N <= burstLen:
            startIdx = 0
        else:
            startIdx = np.random.randint(0, N - burstLen + 1)
        idx = np.arange(startIdx, min(startIdx + burstLen, N))
        llr[idx] = -llr[idx]   # Reverse LLR sign, simulating "extreme distortion"
    
    return llr

def H(p):
    # 二元熵函数，注意边界
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

np.random.seed(2025)  # Fix random seed for reproducibility

# 需要numpy >= 2
# Parameters
K_info = 64  # Watermark information bits
CRClen = 11  # CRC bits (using '11' polynomial)
K_total = K_info + CRClen  # Total length after CRC (Polar input length)

E = 512  # Polar encoding output / rate-matched output length
codeRate = K_info / E  # Effective code rate (only information bits)

L_list = 16  # Polar SCL list length
nMax = 9  # Maximum polarization code length parameter (2^10 = 1024)
iil = True  # Do not use 5G downlink internal interleaving (we do external interleaving)

numFrames = 1  # Number of simulation frames per channel

print(f'Polar audio watermark simulation: K={K_info}, CRC={CRClen}, E={E}, R={codeRate:.3f}, L={L_list}, Frames={numFrames}')
ber_channel = .25  # 25% BER信道
channel_capacity = 1 - H(ber_channel)
print(f'信道容量: C = {channel_capacity}.:4f, 码率 = {codeRate}')
print('✓ 码率 < 信道容量，理论上可能可靠通信' if codeRate < channel_capacity else '✗ 码率 > 信道容量，可靠通信困难')

# 添加 CRC
if CRClen == 6:
    crc_poly = '6'
elif CRClen == 11:
    crc_poly = '11'
else:
    crc_poly = '24C'

# General bit error meter
def errMeterBit(tx, rx):
    return np.sum(tx != rx) / len(tx)

# Main loop: Simulate for multiple channel models
channelTypes = ['BSC_25', 'AWGN_5dB', 'Burst', 'Erasure', 'MP3_like']
numCh = len(channelTypes)

resultsBER = np.zeros(numCh)
resultsFER = np.zeros(numCh)

for ci in range(numCh):
    chName = channelTypes[ci]
    bitErrs = 0
    frmErrs = 0
    totalBits = 0
    
    print(f'\n=== Channel type: {chName} ===\n')
    
    for frm in range(numFrames):
        # 1) Generate watermark bits
        infoBits = np.random.randint(0, 2, K_info)  # Watermark content
        # infoBits = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0,
        #             0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])

        # 2) CRC encoding (11 bit)
        msgCRC = nrCRCEncode(infoBits, crc_poly)  # Length K_total
        msgCRC = msgCRC.flatten()  # Ensure 1D column vector for Polar encode

        # Debug print
        #print(f"msgCRC shape: {msgCRC.shape}, dtype: {msgCRC.dtype}")
        #print(f"msgCRC min: {msgCRC.min()}, max: {msgCRC.max()}, unique: {np.unique(msgCRC)}")
        
        # 3) Polar encoding (E bits output)
        encBits = nrPolarEncode(msgCRC, E, nMax, iil)  # encBits: E x 1 (0/1)
        encBits = encBits.astype(np.float64)

        llrPerfect = (1 - 2* encBits) * 8
        decPerfect = nrPolarDecode(llrPerfect, K_total, E, L_list, nMax, iil, CRClen)
        infoPerfect, crcErrPerfect = nrCRCDecode(decPerfect, crc_poly)

        if crcErrPerfect != 0 or any(infoPerfect.ravel() != infoBits):
            print('Perfect channel test FAILED with sign (1-2*b)*8, try flipping sign!')
        else:
            print('Perfect channel test PASSED with sign (1-2*b)*8.')
        
        # 4) External interleaving
        # encInt = encBits[perm]
        encInt = encBits  # Comment suggests optional, but script uses it; adjust if needed
        
        # 5) Through "audio equivalent channel" -> Get LLR
        if chName == 'BSC_25':
            pFlip = ber_channel  # 25% bit flip
            llrMag = np.log((1 - pFlip) / pFlip)  # BSC theoretical LLR magnitude
            llrInt = bscChannelLLR(encInt, pFlip, llrMag)
            
        elif chName == 'AWGN_5dB':
            EbN0dB = 5  # Rough SNR
            llrInt = awgnChannelLLR(encInt, EbN0dB, codeRate)
            
        elif chName == 'Burst':
            # Intentionally concentrate errors into bursts, simulating compression causing severe distortion in segments
            pFlip = 0.25
            burstLen = round(0.05 * E)  # Each burst covers 5% of codeword
            llrMag = 3  # Confidence magnitude
            llrInt = burstChannelLLR(encInt, pFlip, burstLen, llrMag)
            
        elif chName == 'Erasure':
            # Some bits completely "disappear" or are very weak, LLR=0 means completely uncertain
            erasureFraction = 0.2  # 20% erasure
            baseLLR = 4
            llrInt = erasureChannelLLR(encInt, erasureFraction, baseLLR)
            
        elif chName == 'MP3_like':
            # Rough simulation: AWGN + some erasures + some burst symbol reversals
            llrInt = mp3LikeChannel(encInt, codeRate)
            
        else:
            raise ValueError('Unknown channel type')
        
        # 6) Deinterleaving (restore Polar decoding input order)
        # llrDeint = np.zeros(E)
        # llrDeint[invPerm] = llrInt  # Inverse permutation
        llrDeint = llrInt  # Matching the script's comment
        
        # 7) Polar decoding (SCL + CRC)
        # nrPolarDecode's first output is K_total bits including CRC
        rxBitsCRC = nrPolarDecode(llrDeint, K_total, E, L_list, nMax, iil, CRClen)
        
        # 8) CRC decoding, remove CRC
        rxInfoBits, crcErr = nrCRCDecode(rxBitsCRC, crc_poly)
        rxInfoBits = rxInfoBits.flatten()  # Ensure 1D for comparison
        
        # 9) Statistics bit errors
        bitErrNow = np.sum(rxInfoBits != infoBits)
        bitErrs += bitErrNow
        totalBits += K_info
        
        if bitErrNow > 0 or crcErr != 0:
            frmErrs += 1
    
    resultsBER[ci] = bitErrs / totalBits
    resultsFER[ci] = frmErrs / numFrames
    
    print(f'Channel={chName}: BER={resultsBER[ci]:.4g}, FER={resultsFER[ci]:.4g}')

print('\n=== Simulation ended ===\n')

# Display results table (simple print)
print('{:<16} {:>10} {:>10}'.format('Channel', 'BER', 'FER'))
for i in range(numCh):
    print('{:<16} {:>10.4g} {:>10.4g}'.format(channelTypes[i], resultsBER[i], resultsFER[i]))
