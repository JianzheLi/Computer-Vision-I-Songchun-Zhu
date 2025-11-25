# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, fabs, sqrt
from libc.stdlib cimport rand, RAND_MAX, malloc, free, srand
from cython.parallel import prange, parallel
cimport cython
cimport openmp

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.int32_t INT_t

cnp.import_array()

# 声明 rand_r 函数
cdef extern from "stdlib.h" nogil:
    int rand_r(unsigned int *seedp)

# 简单的线程安全随机数生成器
@cython.cdivision(True)
cdef inline unsigned int xorshift32(unsigned int* state) nogil:
    """快速的线程安全随机数生成器"""
    cdef unsigned int x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline DTYPE_t compute_response_fast(
    DTYPE_t* img,
    DTYPE_t* filter_data,
    int pos_h, int pos_w,
    int H, int W,
    int fh, int fw,
    int half_h, int half_w) noexcept nogil:
    """优化的响应计算 - 使用指针直接访问"""
    cdef DTYPE_t response = 0.0
    cdef int kh, kw, img_h, img_w, img_idx
    
    for kh in range(fh):
        for kw in range(fw):
            img_h = (pos_h - half_h + kh)
            if img_h < 0:
                img_h += H
            elif img_h >= H:
                img_h -= H
            
            img_w = (pos_w - half_w + kw)
            if img_w < 0:
                img_w += W
            elif img_w >= W:
                img_w -= W
            
            img_idx = img_h * W + img_w
            response += img[img_idx] * filter_data[kh * fw + kw]
    
    return response

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int get_bin_index_fast(
    DTYPE_t response,
    DTYPE_t min_resp,
    DTYPE_t max_resp,
    DTYPE_t bin_width,
    int num_bins) noexcept nogil:
    """优化的bin索引计算 - 预计算bin_width"""
    cdef int bin_idx
    
    if response < min_resp or response > max_resp:
        return -1
    
    bin_idx = <int>((response - min_resp) / bin_width)
    
    if bin_idx < 0:
        return 0
    if bin_idx >= num_bins:
        return num_bins - 1
    
    return bin_idx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void pos_gibbs_sample_update_optimized(
    DTYPE_t* img_syn,
    DTYPE_t* hists_syn,
    DTYPE_t* hists_ori,
    DTYPE_t** filters,
    int* filter_sizes,
    DTYPE_t* bounds,
    DTYPE_t* weight,
    DTYPE_t* bin_widths,
    int pos_h, int pos_w,
    int H, int W,
    int num_filters,
    int num_bins,
    DTYPE_t T,
    DTYPE_t norm_factor,
    unsigned int* rand_seed) noexcept nogil:
    """优化的单像素Gibbs采样更新"""
    
    cdef int pixel_idx = pos_h * W + pos_w
    cdef DTYPE_t original_intensity = img_syn[pixel_idx]
    
    # 栈上分配小数组
    cdef DTYPE_t energy[8]
    cdef DTYPE_t probs[8]
    cdef int intensity, f_idx, dh, dw, ph, pw
    cdef int fh, fw, half_h, half_w
    cdef int old_bin, new_bin, kh, kw
    cdef DTYPE_t delta_intensity, old_response, new_response
    cdef DTYPE_t min_resp, max_resp
    cdef DTYPE_t delta_energy, old_diff, new_diff
    cdef DTYPE_t* current_filter
    cdef int img_h, img_w, img_idx
    cdef DTYPE_t filter_val
    
    # 计算每个强度值的能量
    for intensity in range(8):
        delta_intensity = intensity - original_intensity
        
        if fabs(delta_intensity) < 1e-10:
            energy[intensity] = 0.0
            continue
        
        delta_energy = 0.0
        
        # 遍历每个滤波器
        for f_idx in range(num_filters):
            current_filter = filters[f_idx]
            fh = filter_sizes[f_idx * 2]
            fw = filter_sizes[f_idx * 2 + 1]
            half_h = fh >> 1  # 位运算优化除法
            half_w = fw >> 1
            
            min_resp = bounds[f_idx * 2 + 1]
            max_resp = bounds[f_idx * 2]
            
            # 遍历受影响的位置
            for dh in range(-half_h, half_h + 1):
                ph = pos_h + dh
                if ph < 0:
                    ph += H
                elif ph >= H:
                    ph -= H
                
                for dw in range(-half_w, half_w + 1):
                    pw = pos_w + dw
                    if pw < 0:
                        pw += W
                    elif pw >= W:
                        pw -= W
                    
                    # 计算旧响应
                    old_response = 0.0
                    for kh in range(fh):
                        for kw in range(fw):
                            img_h = (ph - half_h + kh)
                            if img_h < 0:
                                img_h += H
                            elif img_h >= H:
                                img_h -= H
                            
                            img_w = (pw - half_w + kw)
                            if img_w < 0:
                                img_w += W
                            elif img_w >= W:
                                img_w -= W
                            
                            img_idx = img_h * W + img_w
                            old_response += img_syn[img_idx] * current_filter[kh * fw + kw]
                    
                    # 计算新响应
                    filter_val = current_filter[(half_h - dh) * fw + (half_w - dw)]
                    new_response = old_response + delta_intensity * filter_val
                    
                    # 获取bin索引
                    old_bin = get_bin_index_fast(old_response, min_resp, max_resp,
                                                 bin_widths[f_idx], num_bins)
                    new_bin = get_bin_index_fast(new_response, min_resp, max_resp,
                                                 bin_widths[f_idx], num_bins)
                    
                    # 更新能量
                    if old_bin >= 0 and new_bin >= 0 and old_bin != new_bin:
                        # 旧bin贡献
                        old_diff = fabs(hists_syn[f_idx * num_bins + old_bin] - 
                                      hists_ori[f_idx * num_bins + old_bin])
                        new_diff = fabs(hists_syn[f_idx * num_bins + old_bin] - norm_factor - 
                                      hists_ori[f_idx * num_bins + old_bin])
                        delta_energy += (new_diff - old_diff) * weight[old_bin]
                        
                        # 新bin贡献
                        old_diff = fabs(hists_syn[f_idx * num_bins + new_bin] - 
                                      hists_ori[f_idx * num_bins + new_bin])
                        new_diff = fabs(hists_syn[f_idx * num_bins + new_bin] + norm_factor - 
                                      hists_ori[f_idx * num_bins + new_bin])
                        delta_energy += (new_diff - old_diff) * weight[new_bin]
        
        energy[intensity] = delta_energy
    
    # 计算概率
    cdef DTYPE_t max_energy = energy[0]
    for intensity in range(1, 8):
        if energy[intensity] > max_energy:
            max_energy = energy[intensity]
    
    cdef DTYPE_t prob_sum = 0.0
    for intensity in range(8):
        probs[intensity] = exp(-(energy[intensity] - max_energy) / T)
        prob_sum += probs[intensity]
    
    # 归一化概率
    cdef DTYPE_t inv_prob_sum = 1.0 / prob_sum
    for intensity in range(8):
        probs[intensity] *= inv_prob_sum
    
    # 采样新强度 - 使用xorshift随机数生成器
    cdef unsigned int rand_int = xorshift32(rand_seed)
    cdef DTYPE_t rand_val = (rand_int % 10000) / 10000.0
    cdef DTYPE_t cdf = 0.0
    cdef int new_intensity = 7
    
    for intensity in range(8):
        cdf += probs[intensity]
        if rand_val <= cdf:
            new_intensity = intensity
            break
    
    # 更新图像
    img_syn[pixel_idx] = new_intensity
    delta_intensity = new_intensity - original_intensity
    
    # 更新直方图
    if fabs(delta_intensity) > 1e-10:
        for f_idx in range(num_filters):
            current_filter = filters[f_idx]
            fh = filter_sizes[f_idx * 2]
            fw = filter_sizes[f_idx * 2 + 1]
            half_h = fh >> 1
            half_w = fw >> 1
            
            min_resp = bounds[f_idx * 2 + 1]
            max_resp = bounds[f_idx * 2]
            
            for dh in range(-half_h, half_h + 1):
                ph = pos_h + dh
                if ph < 0:
                    ph += H
                elif ph >= H:
                    ph -= H
                
                for dw in range(-half_w, half_w + 1):
                    pw = pos_w + dw
                    if pw < 0:
                        pw += W
                    elif pw >= W:
                        pw -= W
                    
                    # 计算旧响应(使用原始强度)
                    old_response = 0.0
                    for kh in range(fh):
                        for kw in range(fw):
                            img_h = (ph - half_h + kh)
                            if img_h < 0:
                                img_h += H
                            elif img_h >= H:
                                img_h -= H
                            
                            img_w = (pw - half_w + kw)
                            if img_w < 0:
                                img_w += W
                            elif img_w >= W:
                                img_w -= W
                            
                            img_idx = img_h * W + img_w
                            if img_h == pos_h and img_w == pos_w:
                                old_response += original_intensity * current_filter[kh * fw + kw]
                            else:
                                old_response += img_syn[img_idx] * current_filter[kh * fw + kw]
                    
                    # 计算新响应
                    filter_val = current_filter[(half_h - dh) * fw + (half_w - dw)]
                    new_response = old_response + delta_intensity * filter_val
                    
                    # 更新直方图
                    old_bin = get_bin_index_fast(old_response, min_resp, max_resp,
                                                 bin_widths[f_idx], num_bins)
                    new_bin = get_bin_index_fast(new_response, min_resp, max_resp,
                                                 bin_widths[f_idx], num_bins)
                    
                    if old_bin >= 0:
                        hists_syn[f_idx * num_bins + old_bin] -= norm_factor
                    if new_bin >= 0:
                        hists_syn[f_idx * num_bins + new_bin] += norm_factor

@cython.boundscheck(False)
@cython.wraparound(False)
def gibbs_sample_cy_optimized(
    DTYPE_t[:, :] img_syn,
    DTYPE_t[:, :] hists_syn,
    DTYPE_t[:, :] hists_ori,
    list filter_list,
    int sweep,
    DTYPE_t[:, :] bounds,
    DTYPE_t T,
    DTYPE_t[:] weight,
    int num_bins,
    int num_threads=0):
    """
    高度优化的Cython Gibbs采样器
    使用OpenMP多线程并行 + 棋盘模式更新
    """
    cdef int H = img_syn.shape[0]
    cdef int W = img_syn.shape[1]
    cdef int num_filters = len(filter_list)
    cdef int s, pos_h, pos_w, f_idx, bin_idx, i
    cdef DTYPE_t error, max_error, bin_error
    cdef DTYPE_t current_T = T
    cdef DTYPE_t norm_factor = 1.0 / (H * W)
    
    # 设置OpenMP线程数
    if num_threads <= 0:
        num_threads = openmp.omp_get_max_threads()
    openmp.omp_set_num_threads(num_threads)
    
    # 预分配和预处理filter数据
    cdef DTYPE_t** filters = <DTYPE_t**>malloc(num_filters * sizeof(DTYPE_t*))
    cdef int* filter_sizes = <int*>malloc(num_filters * 2 * sizeof(int))
    cdef DTYPE_t* bin_widths = <DTYPE_t*>malloc(num_filters * sizeof(DTYPE_t))
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] filter_np
    cdef DTYPE_t[:, :] filter_view
    
    for f_idx in range(num_filters):
        filter_np = filter_list[f_idx]
        filter_view = filter_np
        filters[f_idx] = &filter_view[0, 0]
        filter_sizes[f_idx * 2] = filter_np.shape[0]
        filter_sizes[f_idx * 2 + 1] = filter_np.shape[1]
        bin_widths[f_idx] = (bounds[f_idx, 0] - bounds[f_idx, 1]) / num_bins
    
    # 转换为C数组指针
    cdef DTYPE_t* img_syn_ptr = &img_syn[0, 0]
    cdef DTYPE_t* hists_syn_ptr = &hists_syn[0, 0]
    cdef DTYPE_t* hists_ori_ptr = &hists_ori[0, 0]
    cdef DTYPE_t* bounds_ptr = &bounds[0, 0]
    cdef DTYPE_t* weight_ptr = &weight[0]
    
    # 棋盘模式位置列表
    cdef int num_black = ((H * W) + 1) // 2
    cdef int num_white = (H * W) // 2
    cdef int* black_positions = <int*>malloc(num_black * 2 * sizeof(int))
    cdef int* white_positions = <int*>malloc(num_white * 2 * sizeof(int))
    
    cdef int black_idx = 0
    cdef int white_idx = 0
    
    for pos_h in range(H):
        for pos_w in range(W):
            if (pos_h + pos_w) % 2 == 0:
                black_positions[black_idx * 2] = pos_h
                black_positions[black_idx * 2 + 1] = pos_w
                black_idx += 1
            else:
                white_positions[white_idx * 2] = pos_h
                white_positions[white_idx * 2 + 1] = pos_w
                white_idx += 1
    
    print(f" ---- GIBBS SAMPLING (Cython Optimized with {num_threads} threads) ---- ")
    
    cdef unsigned int rand_seed
    
    for s in range(sweep):
        # 更新黑色格子(并行)
        for i in prange(num_black, nogil=True, schedule='static'):
            rand_seed = i + s * num_black + 12345
            pos_gibbs_sample_update_optimized(
                img_syn_ptr, hists_syn_ptr, hists_ori_ptr,
                filters, filter_sizes, bounds_ptr, weight_ptr, bin_widths,
                black_positions[i * 2], black_positions[i * 2 + 1],
                H, W, num_filters, num_bins, current_T, norm_factor, &rand_seed)
        
        # 更新白色格子(并行)
        for i in prange(num_white, nogil=True, schedule='static'):
            rand_seed = i + s * num_white + num_black + 54321
            pos_gibbs_sample_update_optimized(
                img_syn_ptr, hists_syn_ptr, hists_ori_ptr,
                filters, filter_sizes, bounds_ptr, weight_ptr, bin_widths,
                white_positions[i * 2], white_positions[i * 2 + 1],
                H, W, num_filters, num_bins, current_T, norm_factor, &rand_seed)
        
        # 计算误差
        max_error = 0.0
        error = 0.0
        for f_idx in range(num_filters):
            for bin_idx in range(num_bins):
                bin_error = fabs(hists_syn[f_idx, bin_idx] - hists_ori[f_idx, bin_idx]) * weight[bin_idx]
                error += bin_error
                if bin_error > max_error:
                    max_error = bin_error
        
        error /= num_filters
        print(f'Gibbs iteration {s+1}: error = {error:.6f} max_error: {max_error:.6f}')
        
        # 退火
        current_T *= 0.96
        
        if max_error < 0.1:
            print(f"Gibbs iteration {s+1}: max_error: {max_error:.6f} < 0.1, stop!")
            break
    
    # 释放内存
    free(filters)
    free(filter_sizes)
    free(bin_widths)
    free(black_positions)
    free(white_positions)
    
    return np.asarray(img_syn), np.asarray(hists_syn)