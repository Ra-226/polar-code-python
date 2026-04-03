import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from nrCRC import nrCRCEncode, nrCRCDecode
from nrPolarEncode import nrPolarEncode
from nrPolarDecode import nrPolarDecode
import math

rcParams['font.sans-serif'] = ['PingFang HK'] # 或 ['Heiti SC'], ['Songti SC']
rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def H(p):
    # 二元熵函数，注意边界
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def main():
    # 参数设置 - 调整为更适合高BER场景（尽量忠实于原脚本）
    K_info = 64        # 信息比特数
    crcLen = 11        # CRC 长度
    K_total = K_info + crcLen  # Polar 编码的输入长度

    nMax = 9
    E = 256            # 输出长度
    iIL = True
    L = 16             # 列表长度
    ber_var = 0.2     # 编码后信号在信道上的 BER（10% 示例脚本中使用 0.25）

    # 码率计算
    codeRate = K_info / E
    print('=== %.1f%% BER信道下的Polar码测试 ===' % (ber_var * 100))
    print('列表长度：%d，crcLen：%d' % (L, crcLen))
    print('信息比特: %d, CRC: %d, Polar输入: %d' % (K_info, crcLen, K_total))
    print('码率: %.3f (信息比特/输出长度: %d/%d)' % (codeRate, K_info, E))
    print('目标: 编码后信号在信道中的BER=%.1f%%\n' % (ber_var * 100))

    # 发射端
    print('--- 发射端处理 ---')
    #np.random.seed(42)
    infoBits = np.random.randint(0, 2, K_info)
    print('1. 生成 %d 个信息比特' % K_info)

    # 添加 CRC
    if crcLen == 6:
        crc_poly = '6'
    elif crcLen == 11:
        crc_poly = '11'
    else:
        crc_poly = '24C'
    crcEncoded = nrCRCEncode(infoBits, crc_poly)
    # nrCRCEncode 可能返回二维数组 (K_total,1)，flatten 为 1D
    crcEncoded = np.asarray(crcEncoded).flatten()
    print('2. CRC编码后长度: %d' % (len(crcEncoded)))

    # Polar编码
    polarEncoded = nrPolarEncode(crcEncoded, E, nMax, iIL)
    polarEncoded = np.asarray(polarEncoded).flatten().astype(np.float64)
    print('3. Polar编码后长度: %d' % (len(polarEncoded)))

    # 信道模拟 - 编码后信号BER=10%（脚本中变量 ber_var）
    print('\n--- 信道传输 ---')
    ber_channel = ber_var
    print('编码后信号信道BER: %.2f' % ber_channel)

    # BSC 信道：按概率翻转比特
    rng = np.random.default_rng(42)
    received_hard = polarEncoded.copy()
    error_positions = rng.random(len(received_hard)) < ber_channel
    received_hard[error_positions] = 1 - received_hard[error_positions]

    channel_errors = np.sum(polarEncoded != received_hard)
    channel_ber = channel_errors / len(polarEncoded)
    print('信道实际翻转比特数: %d/%d (实际BER=%.4f)' %
          (int(channel_errors), len(polarEncoded), channel_ber))

    # 接收端处理 - 针对高BER优化
    print('\n--- 接收端处理 ---')

    # 方法1: 标准LLR计算
    print('\n方法1: 标准LLR计算')
    # 防止 division by zero
    if ber_channel <= 0:
        llr_scale = 1e9
    elif ber_channel >= 1:
        llr_scale = -1e9
    else:
        llr_scale = np.log((1 - ber_channel) / ber_channel)
    llr_hard = llr_scale * (1 - 2 * received_hard.astype(float))

    print('LLR尺度因子: %.2f' % llr_scale)

    decodedBits_hard = nrPolarDecode(llr_hard.astype(float), K_total, E, L, nMax, iIL, crcLen)
    decodedBits_hard = np.asarray(decodedBits_hard).flatten().astype(int)
    finalInfoBits_hard = decodedBits_hard[:K_info]

    info_errors_hard = int(np.sum(infoBits != finalInfoBits_hard))
    ber_hard = info_errors_hard / K_info

    print('信息比特错误数: %d/%d' % (info_errors_hard, K_info))
    print('解码后BER: %.4f' % ber_hard)
    if info_errors_hard > 0:
        print('帧错误: 是')
    else:
        print('帧错误: 否')

    # 方法2: 增强LLR尺度（针对高BER）
    print('\n方法2: 增强LLR尺度')
    enhanced_scale = 2.0 * llr_scale
    llr_enhanced = enhanced_scale * (1 - 2 * received_hard.astype(float))

    print('增强LLR尺度因子: %.2f' % enhanced_scale)

    decodedBits_enhanced = nrPolarDecode(llr_enhanced.astype(float), K_total, E, L, nMax, iIL, crcLen)
    decodedBits_enhanced = np.asarray(decodedBits_enhanced).flatten().astype(int)
    finalInfoBits_enhanced = decodedBits_enhanced[:K_info]

    info_errors_enhanced = int(np.sum(infoBits != finalInfoBits_enhanced))
    ber_enhanced = info_errors_enhanced / K_info

    print('信息比特错误数: %d/%d' % (info_errors_enhanced, K_info))
    print('解码后BER: %.4f' % ber_enhanced)
    if info_errors_enhanced > 0:
        print('帧错误: 是')
    else:
        print('帧错误: 否')

    # LLR尺度敏感性测试 - 扩展范围
    print('\n--- LLR尺度敏感性测试 (%.1f%% BER) ---' % (ber_var * 100))
    scales = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    ber_scales = np.zeros(len(scales))

    for i in range(len(scales)):
        scale = scales[i]
        llr_test = scale * (1 - 2 * received_hard.astype(float))
        decoded_test = nrPolarDecode(llr_test.astype(float), K_total, E, L, nMax, iIL, crcLen)
        decoded_test = np.asarray(decoded_test).flatten().astype(int)
        info_test = decoded_test[:K_info]
        ber_test = np.sum(infoBits != info_test) / K_info
        ber_scales[i] = ber_test
        print('LLR尺度=%.1f: BER=%.4f' % (scale, ber_test))

    # 找到最佳尺度
    best_idx = int(np.argmin(ber_scales))
    best_ber = float(ber_scales[best_idx])
    print('最佳LLR尺度: %.1f, 对应BER: %.4f' % (scales[best_idx], best_ber))

    # 不同列表长度性能比较
    print('\n--- 不同列表长度性能 (%.1f%% BER) ---' % (ber_var * 100))
    L_list = [1, 2, 4, 8, 16, 32]
    ber_L = np.zeros(len(L_list))

    for i in range(len(L_list)):
        current_L = L_list[i]
        decoded_L = nrPolarDecode(llr_hard.astype(float), K_total, E, current_L, nMax, iIL, crcLen)
        decoded_L = np.asarray(decoded_L).flatten().astype(int)
        info_L = decoded_L[:K_info]
        ber_L[i] = np.sum(infoBits != info_L) / K_info
        print('列表长度 L=%2d: BER=%.4f' % (current_L, ber_L[i]))

    # 不同码率在指定BER下的性能
    print('\n=== 不同码率在%.1f%% BER下的性能 ===' % (ber_var * 100))
    rates = [0.1, 0.125, 0.15, 0.2, 0.25]
    rate_results = np.zeros((len(rates), 3))  # [码率, E, 解码BER]

    for i in range(len(rates)):
        current_rate = rates[i]
        E_test = int(round(K_info / current_rate))

        # 重新编码（注意：保持与原脚本一致，使用同一 crcEncoded）
        polar_test = nrPolarEncode(crcEncoded, E_test, nMax, iIL)
        polar_test = np.asarray(polar_test).flatten().astype(int)

        # 10% BSC 信道
        received_test = polar_test.copy()
        error_pos = rng.random(len(received_test)) < ber_var
        received_test[error_pos] = 1 - received_test[error_pos]

        # 解码
        llr_scale_test = np.log((1 - ber_var) / ber_var)
        llr_test = llr_scale_test * (1 - 2 * received_test.astype(float))
        decoded_test = nrPolarDecode(llr_test.astype(float), K_total, E_test, 16, nMax, iIL, crcLen)
        decoded_test = np.asarray(decoded_test).flatten().astype(int)
        info_test = decoded_test[:K_info]

        decoded_ber = np.sum(infoBits != info_test) / K_info
        rate_results[i, :] = [current_rate, E_test, decoded_ber]

        print('码率=%.3f (E=%4d): 解码BER=%.4f' % (current_rate, E_test, decoded_ber))

    # 绘制综合性能图
    # plt.figure(figsize=(12, 8))
    #
    # # 子图1: LLR尺度敏感性
    # ax1 = plt.subplot(2, 2, 1)
    # ax1.semilogy(scales, ber_scales, 'bo-', linewidth=2, markersize=8)
    # ax1.semilogy(scales[best_idx], best_ber, 'ro', markersize=10, linewidth=3)
    # ax1.set_xlabel('LLR尺度因子')
    # ax1.set_ylabel('解码BER')
    # ax1.set_title('LLR尺度敏感性 (10% BER)')
    # ax1.grid(True)
    # ax1.legend(['性能', '最佳点'], loc='best')

    # # 子图2: 列表长度影响
    # ax2 = plt.subplot(2, 2, 2)
    # ax2.semilogy(L_list, ber_L, 'gs-', linewidth=2, markersize=8)
    # ax2.set_xlabel('列表长度 L')
    # ax2.set_ylabel('解码BER')
    # ax2.set_title('列表长度对性能的影响')
    # ax2.grid(True)
    #
    # # 子图3: 码率性能
    # ax3 = plt.subplot(2, 2, 3)
    # ax3.plot(rate_results[:, 0], rate_results[:, 2], 'md-', linewidth=2, markersize=8)
    # ax3.set_xlabel('码率')
    # ax3.set_ylabel('解码BER')
    # ax3.set_title('不同码率在10% BER下的性能')
    # ax3.grid(True)
    #
    # # 子图4: 性能总结
    # ax4 = plt.subplot(2, 2, 4)
    # performance_data = np.array([ber_hard, ber_enhanced, best_ber, np.min(ber_L), np.min(rate_results[:, 2])])
    # methods = ['标准', '增强LLR', '最优LLR', '最优L', '最优码率']
    # ax4.bar(range(len(performance_data)), performance_data * 100)
    # ax4.set_xticks(range(len(performance_data)))
    # ax4.set_xticklabels(methods)
    # ax4.set_ylabel('解码BER (%)')
    # ax4.set_title('各种优化方法性能对比')
    # ax4.grid(True)
    # for i in range(len(performance_data)):
    #     ax4.text(i, performance_data[i] * 100 + 0.1, '%.2f%%' % (performance_data[i] * 100),
    #              horizontalalignment='center')
    #
    # plt.tight_layout()
    # plt.show()

    # 理论分析
    print('\n=== 理论分析 (%.1f%%编码后BER) ===' % (ber_var * 100))
    print('当前配置分析:')
    print('- 编码后BER: %.1f%%' % (ber_channel * 100))
    print('- 码率: %.3f' % codeRate)
    if ber_channel > 0 and ber_channel < 1:
        channel_capacity = 1 - H(ber_channel)
        print('- 等效原始信道BER: %.1f%%' % ((ber_channel / codeRate) * 100))
        print('信道容量: C = %.4f' % channel_capacity)
        if codeRate < channel_capacity:
            print('码率 < 信道容量，理论上可能可靠通信')
        else:
            print('码率 > 信道容量，可靠通信困难')

    print('\n优化建议:')
    print('1. 使用较低码率 (<0.15)')
    print('2. 增大列表长度 (L=16~32)')
    print('3. 优化LLR尺度因子')
    print('4. 考虑使用更强大的CRC')

if __name__ == "__main__":
    main()
