import numpy as np


def subBlockInterleave(in_):
    # Sub-block interleaver
    #   OUT = subBlockInterleave(IN) returns the sub-block interleaved output.
    #
    #   Reference: TS 38.212, Section 5.4.1.1.

    N = len(in_)
    jn = subblockInterleaveMap(N)
    out = in_[jn]

    return out


def iBILInterl(in_):
    # Triangular interleaver
    #
    #   OUT = iBILInterl(IN) performs triangular interleaving on the input, IN,
    #   writing in the input E elements row-wise and returns the output, OUT,
    #   by reading them out column-wise.
    #
    #   Reference: TS 38.212, Section 5.4.1.3.

    # Get T off E
    E = len(in_)
    T = getT(E)

    # Write input to buffer row-wise
    v = -1 * np.ones((T, T), dtype=in_.dtype)  # <NULL> bits
    k = 0
    for i in range(T):
        for j in range(T - i):
            if k < E:
                v[i, j] = in_[k]
            k += 1

    # Read output from buffer column-wise
    out = np.zeros(E, dtype=in_.dtype)
    k = 0
    for j in range(T):
        for i in range(T - j):
            if v[i, j] != -1:
                out[k] = v[i, j]
                k += 1

    return out


def getT(E):
    # Use quadratic solution with ceil for >= in expression.
    t = int(np.ceil((-1 + np.sqrt(1 + 8 * E)) / 2))

    return t


def subblockInterleaveMap(N):
    # subblockInterleaveMap Subblock interleaving pattern for Polar rate-matching
    #
    #   out = subblockInterleaveMap(N) returns the
    #   sub-block interleaving pattern for length N.
    #
    #   See also nrRateMatchPolar, nrRateRecoverPolar.
    #

    #
    #   Reference:
    #   [1] 3GPP TS 38.212, "3rd Generation Partnership Project; Technical
    #   Specification Group Radio Access Network; NR; Multiplexing and channel
    #   coding (Release 15). Section 5.4.1.1.

    # Table 5.4.1.1-1: Sub-block interleaver pattern
    pi = np.array([
        0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19,
        12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31
    ])

    jn = np.zeros(N, dtype=int)
    for n in range(N):
        i = int(np.floor(32 * n / N))
        jn[n] = pi[i] * (N // 32) + (n % (N // 32))

    return jn


# Helper functions translated from MATLAB

def interleaveMap(K):
    # interleaveMap Interleaver mapping pattern for Polar coding
    #
    #   pi = interleaveMap(K) returns the interleaving
    #   pattern for a length K.
    #
    #   See also nrPolarEncode, nrPolarDecode.
    #

    # Reference:
    #   [1] 3GPP TS 38.212, "3rd Generation Partnership Project; Technical
    #   Specification Group Radio Access Network; NR; Multiplexing and channel
    #   coding (Release 15). Section 5.3.1.1.

    Kilmax = 164
    pat = getPattern()
    pi = np.zeros(K, dtype=int)
    k = 0
    for m in range(Kilmax):
        if pat[m] >= Kilmax - K:
            pi[k] = pat[m] - (Kilmax - K)
            k += 1

    return pi


def getPattern():
    # Table 5.3.1.1-1: Interleaving pattern for Kilmax = 164
    pat = np.array([
        0, 2, 4, 7, 9, 14, 19, 20, 24, 25, 26, 28, 31, 34, 42, 45, 49,
        50, 51, 53, 54, 56, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 72,
        76, 77, 81, 82, 83, 87, 88, 89, 91, 93, 95, 98, 101, 104, 106,
        108, 110, 111, 113, 115, 118, 119, 120, 122, 123, 126, 127, 129,
        132, 134, 138, 139, 140, 1, 3, 5, 8, 10, 15, 21, 27, 29, 32, 35,
        43, 46, 52, 55, 57, 60, 63, 68, 73, 78, 84, 90, 92, 94, 96, 99,
        102, 105, 107, 109, 112, 114, 116, 121, 124, 128, 130, 133, 135,
        141, 6, 11, 16, 22, 30, 33, 36, 44, 47, 64, 74, 79, 85, 97, 100,
        103, 117, 125, 131, 136, 142, 12, 17, 23, 37, 48, 75, 80, 86,
        137, 143, 13, 18, 38, 144, 39, 145, 40, 146, 41, 147, 148, 149,
        150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
        163
    ])

    return pat


def construct(K, E, nMax):
    # construct Polar code construction
    #
    #   F = construct(K,E,NMAX) returns an N-bit vector, F,
    #   as the output where K entries in the output would be 0 (information bit
    #   locations), and N-K entries in the output would be 1 (frozen bit
    #   locations). E is the rate-matched output length and NMAX is the maximum
    #   n value (either of 9 or 10). The mother code rate is given by K/N,
    #   while the effective code rate after rate-matching is K/E. K, E and NMAX
    #   must be all scalars.
    #
    #   [F,QPC,NPCWM] = construct(K,E,NMAX) also outputs
    #   the set of bit indices for parity check bits QPC and the number of
    #   parity check bits of minimum row weight NPCWM, for K valued in the
    #   range 18<=K<=25.
    #
    #   See also nrPolarEncode, nrPolarDecode.
    #
    # References:
    #   [1] 3GPP TS 38.212, "3rd Generation Partnership Project; Technical
    #   Specification Group Radio Access Network; NR; Multiplexing and channel
    #   coding (Release 15). Section 5.3.1.2.

    # Get N, Section 5.3.1
    N = getN(K, E, nMax)

    # Check and set PC-Polar parameters
    if 18 <= K <= 25:  # for PC-Polar, Section 6.3.1.3.1
        nPC = 3
        if E - K > 189:
            nPCwm = 1
        else:
            nPCwm = 0
    else:  # for CA-Polar
        nPC = 0
        nPCwm = 0

    # Get sequence for N, ascending ordered, Section 5.3.1.2
    s10 = sequence()
    idx = s10 < N
    qSeq = s10[idx]  # 0-based

    # Get frozen, information bit indices sets, qF, qI
    jn = subblockInterleaveMap(N)  # 0-based
    qFtmp = np.array([], dtype=int)
    if E < N:
        if K / E <= 7 / 16:  # puncturing
            for i in range(N - E):
                qFtmp = np.append(qFtmp, jn[i])
            if E >= 3 * N / 4:
                uLim = int(np.ceil(3 * N / 4 - E / 2))
                qFtmp = np.append(qFtmp, np.arange(uLim))
            else:
                uLim = int(np.ceil(9 * N / 16 - E / 4))
                qFtmp = np.append(qFtmp, np.arange(uLim))
            qFtmp = np.unique(qFtmp)
        else:  # shortening
            for i in range(E, N):
                qFtmp = np.append(qFtmp, jn[i])

    # Get qI from qFtmp and qSeq
    qI = np.zeros(K + nPC, dtype=int)
    j = 0
    for i in range(1, N + 1):
        ind = qSeq[N - i]  # flip for most reliable
        if ind in qFtmp:
            continue
        qI[j] = ind
        j += 1
        if j == (K + nPC):
            break

    # Form the frozen bit vector
    # For codegen, use lclsetdiff
    qF = lclsetdiff(qSeq, qI)
    F = np.zeros(N, dtype=int)
    F[qF] = np.ones(len(qF), dtype=int)

    # PC-Polar
    qPC = np.zeros(nPC, dtype=int)
    if nPC > 0:
        qPC[:(nPC - nPCwm)] = qI[-(nPC - nPCwm):]  # least reliable

        if nPCwm > 0:  # assumes ==1, if >0.
            # Get G, nth Kronecker power of kernel
            n = int(np.log2(N))
            ak0 = np.array([[1, 0], [1, 1]])  # Arikan's kernel
            allG = [None] * n  # Initialize list
            allG[0] = ak0  # Assign first
            for i in range(1, n):
                allG[i] = np.kron(allG[i - 1], ak0)
            G = allG[n - 1]
            wg = np.sum(G, axis=1)  # row weight

            qtildeI = qI[:-(nPC)]
            wt_qtildeI = wg[qtildeI]
            minwt = np.min(wt_qtildeI)  # minimum weight
            allminwtIdx = np.where(wt_qtildeI == minwt)[0]

            # most reliable, minimum row weight is first value
            qPC[nPC - 1] = qtildeI[allminwtIdx[0]]

    return F, qPC, nPCwm


def lclsetdiff(tallA, shortB):
    # Output medC has the values in tallA that are not in shortB.
    # Assumptions:
    #   Both tallA and shortB are columns with unique elements.

    medC = np.zeros(len(tallA) - len(shortB), dtype=int)
    c = 0
    for i in range(len(tallA)):
        tmp = tallA[i]
        found = False
        for j in range(len(shortB)):
            if shortB[j] == tmp:
                found = True
                break
        if not found:
            medC[c] = tmp
            c += 1

    return medC


# Additional helper functions translated from MATLAB

def getN(K, E, nMax):
    # getN Returns N for a given K, E and nMax
    #
    #   N = getN(K,E,NMAX) returns the mother code block
    #   length for the specified number of input bits (K), number of
    #   rate-matched output bits (E) and maximum value of n (NMAX).
    #
    #   See also nrPolarEncode.
    #

    #
    # References:
    #   [1] 3GPP TS 38.212, "3rd Generation Partnership Project; Technical
    #   Specification Group Radio Access Network; NR; Multiplexing and channel
    #   coding (Release 15). Section 5.3.1.

    # Get n, N, Section 5.3.1
    cl2e = int(np.ceil(np.log2(E)))
    if E <= (9 / 8) * 2 ** (cl2e - 1) and K / E < 9 / 16:
        n1 = cl2e - 1
    else:
        n1 = cl2e

    rmin = 1 / 8
    n2 = int(np.ceil(np.log2(K / rmin)))

    nMin = 5
    n = max(min([n1, n2, nMax]), nMin)
    N = 2 ** n

    return N


def sequence():
    # sequence Polar sequence with increasing reliability order
    #
    #   seq = sequence() returns the base 1024-length
    #   sequence, in increasing order of reliability.
    #
    #   See also nrPolarEncode, nrPolarDecode.
    #

    # Table 5.3.1.2-1: Polar sequence, Q
    seq = [0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128, 12, 33, 65, 20, 256, 34, 24, 36, 7, 129, 66, 512, 11,
           40, 68, 130, 19, 13, 48, 14, 72, 257, 21, 132, 35, 258, 26, 513, 80, 37, 25, 22, 136, 260, 264, 38, 514, 96,
           67, 41, 144, 28, 69, 42, 516, 49, 74, 272, 160, 520, 288, 528, 192, 544, 70, 44, 131, 81, 50, 73, 15, 320,
           133, 52, 23, 134, 384, 76, 137, 82, 56, 27, 97, 39, 259, 84, 138, 145, 261, 29, 43, 98, 515, 88, 140, 30,
           146, 71, 262, 265, 161, 576, 45, 100, 640, 51, 148, 46, 75, 266, 273, 517, 104, 162, 53, 193, 152, 77, 164,
           768, 268, 274, 518, 54, 83, 57, 521, 112, 135, 78, 289, 194, 85, 276, 522, 58, 168, 139, 99, 86, 60, 280, 89,
           290, 529, 524, 196, 141, 101, 147, 176, 142, 530, 321, 31, 200, 90, 545, 292, 322, 532, 263, 149, 102, 105,
           304, 296, 163, 92, 47, 267, 385, 546, 324, 208, 386, 150, 153, 165, 106, 55, 328, 536, 577, 548, 113, 154,
           79, 269, 108, 578, 224, 166, 519, 552, 195, 270, 641, 523, 275, 580, 291, 59, 169, 560, 114, 277, 156, 87,
           197, 116, 170, 61, 531, 525, 642, 281, 278, 526, 177, 293, 388, 91, 584, 769, 198, 172, 120, 201, 336, 62,
           282, 143, 103, 178, 294, 93, 644, 202, 592, 323, 392, 297, 770, 107, 180, 151, 209, 284, 648, 94, 204, 298,
           400, 608, 352, 325, 533, 155, 210, 305, 547, 300, 109, 184, 534, 537, 115, 167, 225, 326, 306, 772, 157, 656,
           329, 110, 117, 212, 171, 776, 330, 226, 549, 538, 387, 308, 216, 416, 271, 279, 158, 337, 550, 672, 118, 332,
           579, 540, 389, 173, 121, 553, 199, 784, 179, 228, 338, 312, 704, 390, 174, 554, 581, 393, 283, 122, 448, 353,
           561, 203, 63, 340, 394, 527, 582, 556, 181, 295, 285, 232, 124, 205, 182, 643, 562, 286, 585, 299, 354, 211,
           401, 185, 396, 344, 586, 645, 593, 535, 240, 206, 95, 327, 564, 800, 402, 356, 307, 301, 417, 213, 568, 832,
           588, 186, 646, 404, 227, 896, 594, 418, 302, 649, 771, 360, 539, 111, 331, 214, 309, 188, 449, 217, 408, 609,
           596, 551, 650, 229, 159, 420, 310, 541, 773, 610, 657, 333, 119, 600, 339, 218, 368, 652, 230, 391, 313, 450,
           542, 334, 233, 555, 774, 175, 123, 658, 612, 341, 777, 220, 314, 424, 395, 673, 583, 355, 287, 183, 234, 125,
           557, 660, 616, 342, 316, 241, 778, 563, 345, 452, 397, 403, 207, 674, 558, 785, 432, 357, 187, 236, 664, 624,
           587, 780, 705, 126, 242, 565, 398, 346, 456, 358, 405, 303, 569, 244, 595, 189, 566, 676, 361, 706, 589, 215,
           786, 647, 348, 419, 406, 464, 680, 801, 362, 590, 409, 570, 788, 597, 572, 219, 311, 708, 598, 601, 651, 421,
           792, 802, 611, 602, 410, 231, 688, 653, 248, 369, 190, 364, 654, 659, 335, 480, 315, 221, 370, 613, 422, 425,
           451, 614, 543, 235, 412, 343, 372, 775, 317, 222, 426, 453, 237, 559, 833, 804, 712, 834, 661, 808, 779, 617,
           604, 433, 720, 816, 836, 347, 897, 243, 662, 454, 318, 675, 618, 898, 781, 376, 428, 665, 736, 567, 840, 625,
           238, 359, 457, 399, 787, 591, 678, 434, 677, 349, 245, 458, 666, 620, 363, 127, 191, 782, 407, 436, 626, 571,
           465, 681, 246, 707, 350, 599, 668, 790, 460, 249, 682, 573, 411, 803, 789, 709, 365, 440, 628, 689, 374, 423,
           466, 793, 250, 371, 481, 574, 413, 603, 366, 468, 655, 900, 805, 615, 684, 710, 429, 794, 252, 373, 605, 848,
           690, 713, 632, 482, 806, 427, 904, 414, 223, 663, 692, 835, 619, 472, 455, 796, 809, 714, 721, 837, 716, 864,
           810, 606, 912, 722, 696, 377, 435, 817, 319, 621, 812, 484, 430, 838, 667, 488, 239, 378, 459, 622, 627, 437,
           380, 818, 461, 496, 669, 679, 724, 841, 629, 351, 467, 438, 737, 251, 462, 442, 441, 469, 247, 683, 842, 738,
           899, 670, 783, 849, 820, 728, 928, 791, 367, 901, 630, 685, 844, 633, 711, 253, 691, 824, 902, 686, 740, 850,
           375, 444, 470, 483, 415, 485, 905, 795, 473, 634, 744, 852, 960, 865, 693, 797, 906, 715, 807, 474, 636, 694,
           254, 717, 575, 913, 798, 811, 379, 697, 431, 607, 489, 866, 723, 486, 908, 718, 813, 476, 856, 839, 725, 698,
           914, 752, 868, 819, 814, 439, 929, 490, 623, 671, 739, 916, 463, 843, 381, 497, 930, 821, 726, 961, 872, 492,
           631, 729, 700, 443, 741, 845, 920, 382, 822, 851, 730, 498, 880, 742, 445, 471, 635, 932, 687, 903, 825, 500,
           846, 745, 826, 732, 446, 962, 936, 475, 853, 867, 637, 907, 487, 695, 746, 828, 753, 854, 857, 504, 799, 255,
           964, 909, 719, 477, 915, 638, 748, 944, 869, 491, 699, 754, 858, 478, 968, 383, 910, 815, 976, 870, 917, 727,
           493, 873, 701, 931, 756, 860, 499, 731, 823, 922, 874, 918, 502, 933, 743, 760, 881, 494, 702, 921, 501, 876,
           847, 992, 447, 733, 827, 934, 882, 937, 963, 747, 505, 855, 924, 734, 829, 965, 938, 884, 506, 749, 945, 966,
           755, 859, 940, 830, 911, 871, 639, 888, 479, 946, 750, 969, 508, 861, 757, 970, 919, 875, 862, 758, 948, 977,
           923, 972, 761, 877, 952, 495, 703, 935, 978, 883, 762, 503, 925, 878, 735, 993, 885, 939, 994, 980, 926, 764,
           941, 967, 886, 831, 947, 507, 889, 984, 751, 942, 996, 971, 890, 509, 949, 973, 1000, 892, 950, 863, 759,
           1008, 510, 979, 953, 763, 974, 954, 879, 981, 982, 927, 995, 765, 956, 887, 985, 997, 986, 943, 891, 998,
           766, 511, 988, 1001, 951, 1002, 893, 975, 894, 1009, 955, 1004, 1010, 957, 983, 958, 987, 1012, 999, 1016,
           767, 989, 1003, 990, 1005, 959, 1011, 1013, 895, 1006, 1014, 1017, 1018, 991, 1020, 1007, 1015, 1019, 1021,
           1022, 1023]

    s = [1023, 1022, 767, 991, 1021, 1019, 1015, 1007, 959, 895, 894, 958, 990, 1006, 1014, 511, 893, 957, 989, 1005,
         1013, 766, 891, 955, 987, 1003, 887, 765, 951, 983, 1011, 943, 999, 975, 763, 879, 927, 1020, 1018, 1017, 974,
         510, 950, 759, 998, 982, 863, 892, 509, 942, 973, 949, 997, 886, 981, 941, 751, 995, 507, 890, 971, 831, 926,
         764, 947, 885, 878, 939, 925, 503, 762, 1016, 735, 979, 1004, 935, 877, 967, 761, 923, 883, 956, 758, 862, 495,
         875, 988, 1012, 757, 919, 703, 861, 508, 954, 889, 1002, 985, 871, 750, 755, 953, 1010, 859, 506, 911, 479,
         986, 749, 830, 1001, 505, 1009, 639, 966, 734, 829, 934, 747, 502, 924, 855, 965, 733, 940, 876, 922, 827, 501,
         888, 447, 743, 760, 933, 963, 847, 731, 874, 918, 494, 702, 984, 499, 756, 823, 952, 917, 493, 860, 701, 870,
         1008, 754, 727, 910, 858, 383, 869, 931, 748, 478, 815, 491, 972, 699, 915, 909, 884, 857, 477, 746, 994, 719,
         854, 638, 828, 948, 504, 487, 980, 695, 907, 745, 853, 637, 826, 475, 799, 732, 742, 867, 446, 500, 938, 846,
         255, 825, 851, 753, 730, 970, 903, 741, 635, 445, 687, 822, 845, 882, 498, 471, 729, 1000, 946, 821, 492, 739,
         726, 443, 700, 921, 497, 843, 631, 490, 977, 814, 725, 873, 463, 382, 819, 698, 671, 937, 881, 839, 439, 813,
         489, 381, 718, 993, 476, 945, 697, 486, 723, 623, 969, 978, 694, 996, 474, 798, 717, 485, 811, 379, 908, 431,
         636, 473, 856, 693, 797, 906, 920, 715, 483, 254, 470, 634, 607, 807, 444, 902, 824, 375, 844, 686, 795, 744,
         469, 691, 253, 880, 633, 852, 415, 442, 711, 685, 842, 820, 630, 728, 462, 367, 944, 467, 251, 441, 901, 791,
         752, 629, 818, 683, 575, 670, 968, 496, 724, 461, 438, 838, 812, 872, 380, 936, 669, 622, 627, 437, 722, 837,
         810, 783, 459, 247, 679, 488, 916, 351, 378, 696, 621, 716, 817, 667, 964, 430, 435, 484, 377, 740, 796, 714,
         692, 806, 619, 606, 455, 429, 835, 374, 239, 472, 794, 713, 805, 663, 482, 690, 605, 899, 373, 710, 319, 684,
         414, 427, 468, 868, 793, 615, 689, 252, 481, 632, 962, 790, 709, 603, 413, 682, 366, 655, 371, 803, 466, 574,
         250, 223, 628, 423, 930, 789, 440, 681, 365, 460, 573, 465, 850, 411, 668, 249, 678, 914, 626, 599, 707, 782,
         458, 436, 246, 787, 350, 666, 363, 677, 620, 866, 571, 809, 625, 781, 457, 245, 407, 434, 976, 665, 191, 349,
         738, 618, 591, 454, 675, 428, 662, 841, 359, 929, 433, 376, 238, 779, 567, 617, 721, 243, 453, 661, 347, 604,
         426, 614, 399, 237, 905, 318, 849, 372, 737, 654, 602, 613, 425, 659, 775, 559, 127, 317, 961, 451, 222, 235,
         412, 865, 422, 913, 653, 601, 370, 343, 932, 611, 221, 598, 410, 421, 572, 364, 992, 792, 315, 680, 369, 231,
         651, 788, 597, 409, 248, 808, 543, 335, 570, 362, 219, 419, 406, 190, 480, 590, 595, 311, 688, 569, 361, 244,
         647, 736, 405, 864, 589, 189, 780, 348, 566, 358, 624, 664, 816, 215, 464, 242, 398, 565, 357, 346, 403, 912,
         303, 587, 778, 187, 236, 660, 616, 397, 241, 345, 720, 432, 558, 563, 355, 342, 207, 777, 234, 676, 126, 658,
         316, 612, 583, 557, 395, 774, 341, 183, 424, 652, 233, 600, 125, 848, 287, 456, 904, 314, 610, 220, 230, 773,
         555, 334, 542, 339, 650, 420, 313, 596, 391, 229, 218, 452, 123, 712, 333, 175, 310, 541, 368, 408, 649, 418,
         217, 594, 840, 551, 609, 309, 227, 646, 214, 568, 331, 588, 404, 360, 188, 539, 900, 119, 593, 804, 213, 302,
         645, 307, 564, 402, 356, 159, 586, 186, 327, 301, 396, 344, 206, 111, 535, 771, 211, 562, 401, 185, 354, 585,
         708, 582, 240, 299, 205, 556, 286, 182, 394, 340, 897, 561, 353, 124, 643, 285, 181, 554, 393, 834, 527, 232,
         338, 203, 122, 581, 95, 295, 390, 312, 332, 553, 174, 283, 337, 179, 228, 786, 121, 540, 550, 389, 330, 199,
         308, 173, 674, 118, 216, 579, 226, 538, 549, 898, 450, 329, 279, 306, 117, 63, 706, 158, 171, 212, 387, 326,
         537, 300, 225, 801, 305, 157, 547, 110, 534, 115, 210, 325, 298, 271, 657, 167, 833, 184, 109, 204, 417, 533,
         673, 155, 209, 297, 323, 284, 180, 202, 928, 294, 449, 94, 107, 526, 705, 531, 785, 201, 282, 178, 151, 293,
         93, 802, 525, 198, 172, 836, 960, 103, 281, 177, 560, 120, 291, 278, 197, 91, 170, 143, 523, 352, 62, 552, 277,
         116, 608, 169, 336, 61, 195, 156, 166, 448, 87, 114, 270, 672, 275, 519, 165, 154, 548, 59, 108, 592, 113, 269,
         536, 153, 224, 304, 79, 106, 784, 163, 150, 416, 328, 55, 267, 532, 105, 296, 208, 149, 704, 92, 584, 102, 530,
         292, 142, 101, 90, 147, 200, 47, 263, 776, 524, 141, 89, 280, 176, 400, 99, 86, 290, 196, 60, 522, 656, 139,
         276, 85, 168, 800, 194, 31, 58, 521, 274, 164, 78, 518, 83, 57, 324, 135, 268, 77, 54, 112, 152, 392, 648, 162,
         517, 53, 266, 75, 148, 104, 161, 770, 265, 46, 51, 580, 146, 100, 262, 193, 71, 45, 140, 88, 388, 98, 261, 515,
         43, 30, 138, 84, 97, 29, 769, 137, 82, 259, 642, 39, 134, 56, 76, 81, 27, 145, 133, 322, 52, 74, 546, 73, 50,
         23, 772, 386, 70, 44, 131, 49, 577, 69, 42, 273, 15, 641, 41, 289, 832, 28, 67, 321, 38, 529, 26, 37, 385, 545,
         25, 578, 35, 22, 644, 896, 21, 96, 80, 14, 48, 19, 192, 320, 13, 40, 160, 11, 36, 72, 24, 7, 544, 384, 20, 34,
         18, 576, 12, 144, 10, 288, 9, 6, 528, 272, 5, 520, 136, 68, 640, 264, 132, 516, 3, 260, 514, 66, 130, 258, 513,
         17, 33, 65, 129, 257, 768, 256, 128, 64, 512, 32, 16, 8, 4, 2, 1, 0]
    s = s[::-1]

    seq = np.array(seq)

    return seq
