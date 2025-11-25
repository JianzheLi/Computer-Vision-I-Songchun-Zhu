# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, fabs, log
from libc.stdlib cimport rand, RAND_MAX, malloc, free
from cython.parallel import prange
cimport openmp

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.int32_t INT_t

cnp.import_array()

cdef inline unsigned int xorshift32(unsigned int* state) nogil:
    """快速线程安全随机数生成器"""
    cdef unsigned int x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x

cdef inline DTYPE_t compute_response_circular(
    DTYPE_t* img,
    DTYPE_t* filter_data,
    int pos_h, int pos_w,
    int H, int W,
    int fh, int fw,
    int half_h, int half_w) noexcept nogil:
    """计算滤波器响应（循环边界）"""
    cdef DTYPE_t response = 0.0
    cdef int kh, kw, img_h, img_w
    
    for kh in range(fh):
        for kw in range(fw):
            img_h = (pos_h - half_h + kh + H) % H
            img_w = (pos_w - half_w + kw + W) % W
            response += img[img_h * W + img_w] * filter_data[kh * fw + kw]
    
    return response

cdef inline int get_bin_index(
    DTYPE_t response,
    DTYPE_t min_resp,
    DTYPE_t bin_width,
    int num_bins) noexcept nogil:
    """获取bin索引"""
    cdef int bin_idx = <int>((response - min_resp) / bin_width)
    if bin_idx < 0:
        return 0
    if bin_idx >= num_bins:
        return num_bins - 1
    return bin_idx

cdef void update_pixel_frame(
    DTYPE_t* img_syn,
    DTYPE_t** filters,
    int* filter_sizes,
    DTYPE_t* lambdas,
    DTYPE_t* bin_mins,
    DTYPE_t* bin_widths,
    int pos_h, int pos_w,
    int H, int W,
    int num_filters,
    int num_bins,
    DTYPE_t T,
    unsigned int* rand_seed) noexcept nogil:
    """更新单个像素（FRAME模型）"""
    
    # 声明所有变量在函数开始处
    cdef int pixel_idx = pos_h * W + pos_w
    cdef DTYPE_t original_intensity = img_syn[pixel_idx]
    cdef DTYPE_t energies[8]
    cdef DTYPE_t probs[8]
    cdef int intensity, f_idx, dh, dw
    cdef int fh, fw, half_h, half_w
    cdef int ph, pw
    cdef DTYPE_t delta_intensity
    cdef DTYPE_t old_response, new_response
    cdef int old_bin, new_bin
    cdef DTYPE_t* current_filter
    cdef int kernel_h, kernel_w
    cdef DTYPE_t max_energy, prob_sum, inv_sum
    cdef unsigned int rand_val
    cdef DTYPE_t rand_float, cdf
    cdef int new_intensity
    
    # 对每个候选强度计算能量
    for intensity in range(8):
        energies[intensity] = 0.0
        delta_intensity = intensity - original_intensity
        
        if fabs(delta_intensity) < 1e-10:
            continue
        
        # 计算能量变化
        for f_idx in range(num_filters):
            current_filter = filters[f_idx]
            fh = filter_sizes[f_idx * 2]
            fw = filter_sizes[f_idx * 2 + 1]
            half_h = fh >> 1
            half_w = fw >> 1
            
            # 遍历受影响的响应位置
            for dh in range(-half_h, half_h + 1):
                ph = (pos_h + dh + H) % H
                
                for dw in range(-half_w, half_w + 1):
                    pw = (pos_w + dw + W) % W
                    
                    # 计算旧响应（使用原始强度）
                    old_response = compute_response_circular(
                        img_syn, current_filter, ph, pw,
                        H, W, fh, fw, half_h, half_w)
                    
                    # 计算新响应的增量
                    # 只有当 (ph + offset) = pos 时，响应才会改变
                    kernel_h = half_h - dh
                    kernel_w = half_w - dw
                    if 0 <= kernel_h < fh and 0 <= kernel_w < fw:
                        new_response = old_response + delta_intensity * current_filter[kernel_h * fw + kernel_w]
                    else:
                        new_response = old_response
                    
                    # 计算bin索引
                    old_bin = get_bin_index(old_response, bin_mins[f_idx], bin_widths[f_idx], num_bins)
                    new_bin = get_bin_index(new_response, bin_mins[f_idx], bin_widths[f_idx], num_bins)
                    
                    # 能量贡献：-lambda * phi
                    if old_bin != new_bin:
                        energies[intensity] -= lambdas[f_idx * num_bins + old_bin]
                        energies[intensity] += lambdas[f_idx * num_bins + new_bin]
    
    # 计算概率（数值稳定版本）
    max_energy = energies[0]
    for intensity in range(1, 8):
        if energies[intensity] > max_energy:
            max_energy = energies[intensity]
    
    prob_sum = 0.0
    for intensity in range(8):
        probs[intensity] = exp(-(energies[intensity] - max_energy) / T)
        prob_sum += probs[intensity]
    
    # 归一化
    inv_sum = 1.0 / (prob_sum + 1e-12)
    for intensity in range(8):
        probs[intensity] *= inv_sum
    
    # 采样新强度
    rand_val = xorshift32(rand_seed)
    rand_float = (rand_val % 10000) / 10000.0
    cdf = 0.0
    new_intensity = 7
    
    for intensity in range(8):
        cdf += probs[intensity]
        if rand_float <= cdf:
            new_intensity = intensity
            break
    
    # 更新图像
    img_syn[pixel_idx] = new_intensity

def gibbs_sample_frame_cy(
    DTYPE_t[:, :] img_syn,
    list filter_list,
    DTYPE_t[:, :] lambdas,
    DTYPE_t[:, :] bounds,
    int num_bins,
    int sweep,
    DTYPE_t T,
    int num_threads=0):
    """
    Cython优化的FRAME模型Gibbs采样
    
    参数:
        img_syn: 合成图像 (H, W)
        filter_list: 滤波器列表
        lambdas: Lagrange乘子 (num_filters, num_bins)
        bounds: 每个滤波器的响应范围 (num_filters, 2) - [max, min]
        num_bins: bin数量
        sweep: sweep次数
        T: 初始温度
        num_threads: OpenMP线程数
    """
    cdef int H = img_syn.shape[0]
    cdef int W = img_syn.shape[1]
    cdef int num_filters = len(filter_list)
    cdef int s, i
    cdef DTYPE_t current_T = T
    cdef DTYPE_t rng
    
    # 设置线程数
    if num_threads <= 0:
        num_threads = openmp.omp_get_max_threads()
    openmp.omp_set_num_threads(num_threads)
    
    # 预处理滤波器数据
    cdef DTYPE_t** filters = <DTYPE_t**>malloc(num_filters * sizeof(DTYPE_t*))
    cdef int* filter_sizes = <int*>malloc(num_filters * 2 * sizeof(int))
    cdef DTYPE_t* bin_mins = <DTYPE_t*>malloc(num_filters * sizeof(DTYPE_t))
    cdef DTYPE_t* bin_widths = <DTYPE_t*>malloc(num_filters * sizeof(DTYPE_t))
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] filter_np
    cdef DTYPE_t[:, :] filter_view
    cdef int f_idx
    
    for f_idx in range(num_filters):
        filter_np = filter_list[f_idx]
        filter_view = filter_np
        filters[f_idx] = &filter_view[0, 0]
        filter_sizes[f_idx * 2] = filter_np.shape[0]
        filter_sizes[f_idx * 2 + 1] = filter_np.shape[1]
        
        # bounds: [max, min]
        bin_mins[f_idx] = bounds[f_idx, 1]
        rng = bounds[f_idx, 0] - bounds[f_idx, 1]
        if rng < 1e-6:
            rng = 1e-6
        bin_widths[f_idx] = rng / num_bins
    
    # 转换为C指针
    cdef DTYPE_t* img_syn_ptr = &img_syn[0, 0]
    cdef DTYPE_t* lambdas_ptr = &lambdas[0, 0]
    
    # 棋盘模式位置
    cdef int num_black = ((H * W) + 1) // 2
    cdef int num_white = (H * W) // 2
    cdef int* black_pos = <int*>malloc(num_black * 2 * sizeof(int))
    cdef int* white_pos = <int*>malloc(num_white * 2 * sizeof(int))
    
    cdef int black_idx = 0, white_idx = 0
    cdef int pos_h, pos_w
    
    for pos_h in range(H):
        for pos_w in range(W):
            if (pos_h + pos_w) % 2 == 0:
                black_pos[black_idx * 2] = pos_h
                black_pos[black_idx * 2 + 1] = pos_w
                black_idx += 1
            else:
                white_pos[white_idx * 2] = pos_h
                white_pos[white_idx * 2 + 1] = pos_w
                white_idx += 1
    
    print(f" ---- GIBBS SAMPLING FOR FRAME (Cython with {num_threads} threads) ---- ")
    
    cdef unsigned int rand_seed
    
    # 主循环
    for s in range(sweep):
        # 更新黑色格子（并行）
        for i in prange(num_black, nogil=True, schedule='static'):
            rand_seed = i + s * num_black + 12345
            update_pixel_frame(
                img_syn_ptr, filters, filter_sizes,
                lambdas_ptr, bin_mins, bin_widths,
                black_pos[i * 2], black_pos[i * 2 + 1],
                H, W, num_filters, num_bins, current_T, &rand_seed)
        
        # 更新白色格子（并行）
        for i in prange(num_white, nogil=True, schedule='static'):
            rand_seed = i + s * num_white + num_black + 54321
            update_pixel_frame(
                img_syn_ptr, filters, filter_sizes,
                lambdas_ptr, bin_mins, bin_widths,
                white_pos[i * 2], white_pos[i * 2 + 1],
                H, W, num_filters, num_bins, current_T, &rand_seed)
        
        # 退火
        current_T *= 0.96
        
        if (s + 1) % 10 == 0:
            print(f'Gibbs sweep {s+1}/{sweep} completed, T={current_T:.4f}')
    
    # 释放内存
    free(filters)
    free(filter_sizes)
    free(bin_mins)
    free(bin_widths)
    free(black_pos)
    free(white_pos)
    
    return np.asarray(img_syn)