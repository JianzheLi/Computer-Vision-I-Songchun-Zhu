''' 
This is the main file of Part 1: Julesz Ensemble, julesz.py
'''
#final.py

from numpy.ctypeslib import ndpointer
import numpy as np
from filters import get_filters
import cv2
from torch.nn.functional import conv2d, pad
import torch
from gibbs_optimized import gibbs_sample_cy_optimized
from gibbs import gibbs_sample
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from gibbs import gibbs_sample
import os


def plot_errors(iterations, max_errors, sum_errors, img_name, save_img=True):
    '''
    Plot max and sum errors over iterations
    '''
    plt.figure(figsize=(10, 6))
    
    # 绘制两条曲线
    plt.plot(iterations, max_errors, 'b-o', linewidth=2, markersize=8, label='Max Error')
    plt.plot(iterations, sum_errors, 'r-s', linewidth=2, markersize=8, label='Sum Error')
    
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Error (Average per pixel)', fontsize=14)
    plt.title(f'Histogram Errors over Iterations - {img_name}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(iterations)
    
    plt.tight_layout()
    
    if save_img:
        save_path = f"results/{img_name.split('.')[0]}/error_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error plot saved to {save_path}")
    
    #plt.show()
    plt.close()

def conv(image, filter):
    ''' 
    Computes the filter response on an image.
    Notice: Choose your padding method!
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape (x, x)
    Return:
        filtered_image: numpy array of shape (H, W)
    '''

    # TODO
    # circular padding ("wrap" mode)
    filtered_image = convolve2d(image, filter, mode='same', boundary='wrap')
    #print("filtered image",filtered_image.shape)
    return filtered_image

def get_histogram(filtered_image,bins_num, max_response, min_response, img_H, img_W):
    ''' 
    Computes the normalized filter response histogram on an image.
    Parameters:
        1. filtered_image: numpy array of shape (H, W)
        2. bins_num: int, the number of bins
        3. max_response: int, the maximum response of the filter
        4. min_response: int, the minimum response of the filter
    Return:
        1. histogram: histogram (numpy array)
    '''

    # TODO
    if max_response <= min_response:
        max_response = min_response + 1e-6

    # create bin edges (num_bins equal-length intervals between min and max)
    edges = np.linspace(min_response, max_response,bins_num+1)

    hist, _ = np.histogram(filtered_image.ravel(), bins=edges)
    hist = hist.astype(np.float64) / (img_H * img_W)   # normalized by pixel count
    return hist

    return histogram

def julesz(img_size = 64, img_name = "fur_obs.jpg", save_img = True):
    ''' 
    The main method
    Parameters:
        1. img_size: int, the size of the image
        2. img_name: str, the name of the image
        3. save_img: bool, whether to save intermediate results, for autograder
    '''
    max_intensity = 255
    
    # 获取所有滤波器
    F_list = get_filters()
    F_list = [filter.astype(np.float32) for filter in F_list]
    
    # 选中的滤波器列表（初始为空）
    filter_list = []
    
    # 图像尺寸
    img_H = img_W = img_size
    
    # 读取原始图像
    img_ori = cv2.resize(cv2.imread(f'images/{img_name}', cv2.IMREAD_GRAYSCALE), (img_H, img_W))
    img_ori = (img_ori).astype(np.float32)
    img_ori = img_ori * 7 // max_intensity  # 映射到 [0, 7]
    

    if save_img:
        save_dir = f"results/{img_name.split('.')[0]}"
        os.makedirs(save_dir, exist_ok=True)  


    # 保存原始图像


    if save_img:
        cv2.imwrite(f"results/{img_name.split('.')[0]}/original.jpg", (img_ori / 7 * 255))
    
    # 从随机噪声初始化合成图像
    img_syn = np.random.randint(0, 8, img_ori.shape).astype(np.float32)
    
    # ==========================================
    # 超参数设置
    # ==========================================
    num_bins = 15  # 直方图的 bin 数量
    threshold = 0.1  # 停止阈值（根据图像大小调整）
    max_error = float('inf')  # 初始设为无穷大
    max_iterations = 50  # 最大迭代次数（防止无限循环）
    
    # Gibbs 采样参数
    gibbs_sweeps = 150  # 每次迭代的 Gibbs sweep 次数
    initial_temperature = 1
    
    # 权重函数（给每个 bin 的权重）
    weight = np.array([8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8])
    #weight = np.ones(num_bins)  # 可以调整为非均匀权重
    
    # ==========================================
    # 预计算：原始图像的所有滤波器响应和边界
    # ==========================================
    print("Computing filter responses on original image...")
    ori_responses = []
    bounds = []  # 每个滤波器的 (max_response, min_response)

    iterations = []
    max_errors = []
    sum_errors = []
    for f in F_list:
        resp = conv(img_ori, f)
        ori_responses.append(resp)
        max_resp = np.max(resp)
        min_resp = np.min(resp)
        bounds.append((max_resp, min_resp))
    
    # ==========================================
    # 主循环：逐步选择滤波器
    # ==========================================
    round = 0
    print("---- Julesz Ensemble Synthesis ----")
    
    while max_error > threshold and round < max_iterations:
        print(f"\n=== Round {round + 1} ===")
        
        # ==========================================
        # Step 4: 对每个未选中的滤波器计算误差
        # ==========================================
        filter_errors = []
        
        for idx, f in enumerate(F_list):
            # 修复：使用正确的方式检查滤波器是否已选中
            is_selected = False
            for selected in filter_list:
                if np.array_equal(f, selected):
                    is_selected = True
                    break
            
            if is_selected:
                filter_errors.append(-1)
                continue
            
            # 计算原始图像的响应直方图
            ori_resp = ori_responses[idx]
            max_resp, min_resp = bounds[idx]
            hist_ori = get_histogram(ori_resp, num_bins, max_resp, min_resp, img_H, img_W)
            
            # 计算合成图像的响应直方图（使用相同的 bin 范围）
            syn_resp = conv(img_syn, f)
            hist_syn = get_histogram(syn_resp, num_bins, max_resp, min_resp, img_H, img_W)
            
            # 计算加权误差
            error = np.sum(np.abs(hist_ori - hist_syn) * weight) * (img_H * img_W)
            filter_errors.append(error)
        
        # ==========================================
        # Step 5: 选择误差最大的滤波器
        # ==========================================
        max_error = max(filter_errors)
        
        if max_error <= threshold*(img_H * img_W):
            print(f"Max error {max_error:.4f} <= threshold {threshold}, stopping.")
            break
        
        # 选择误差最大的滤波器
        max_error_idx = np.argmax(filter_errors)
        selected_filter = F_list[max_error_idx]
        filter_list.append(selected_filter)
        
        print(f"Selected filter #{max_error_idx} with error {max_error:.4f}")
        print(f"Total selected filters: {len(filter_list)}")
        
        # ==========================================
        # Step 6: Gibbs 采样更新图像
        # ==========================================
        # 准备所有已选滤波器的直方图和边界
        selected_hists_ori = []
        selected_hists_syn = []
        selected_bounds = []

        for f in filter_list:
            # 找到这个滤波器在 F_list 中的索引
            f_idx = None
            for i, ff in enumerate(F_list):
                if np.array_equal(f, ff):
                    f_idx = i
                    break
            
            if f_idx is not None:
                # 原始图像的直方图
                ori_resp = ori_responses[f_idx]
                max_resp, min_resp = bounds[f_idx]
                hist_ori = get_histogram(ori_resp, num_bins, max_resp, min_resp, img_H, img_W)
                selected_hists_ori.append(hist_ori)
                
                # 合成图像的直方图
                syn_resp = conv(img_syn, f)
                hist_syn = get_histogram(syn_resp, num_bins, max_resp, min_resp, img_H, img_W)
                selected_hists_syn.append(hist_syn)
                
                # 边界 - 注意顺序：(max, min)
                selected_bounds.append([max_resp, min_resp])

        # 转换为 numpy 数组，确保类型正确
        hists_ori_array = np.array(selected_hists_ori, dtype=np.float32)
        hists_syn_array = np.array(selected_hists_syn, dtype=np.float32)
        bounds_array = np.array(selected_bounds, dtype=np.float32)

        # 确保 filter_list 中的滤波器都是 float32 类型
        filter_list_float32 = [f.astype(np.float32) for f in filter_list]

        # 确保其他参数类型正确
        weight_float32 = weight.astype(np.float32) if not isinstance(weight, np.ndarray) or weight.dtype != np.float32 else weight
        img_syn = img_syn.astype(np.float32)

        # 运行 Gibbs 采样
        print(f"Running Gibbs sampling ({gibbs_sweeps} sweeps)...")
        """img_syn, hists_syn_array = gibbs_sample(
            img_syn,
            hists_syn_array,      # ← 用所有 selected filters 的直方图
            img_ori,
            hists_ori_array,      # ← 原图对应的所有直方图
            filter_list,
            gibbs_sweeps,
            bounds_array,         # ← 每个滤波器对应的 (max, min)
            initial_temperature,
            weight,
            num_bins
        )"""

        img_syn, hists_syn_array = gibbs_sample_cy_optimized(  
            img_syn, 
            hists_syn_array,
            hists_ori_array,
            filter_list_float32,
            gibbs_sweeps,  # 注意：这里直接传递值，不是关键字参数
            bounds_array,
            initial_temperature,
            weight_float32,
            num_bins,
            8  # num_threads
        )
        # ==========================================
        # Step 7: 计算总误差
        # ==========================================
        total_error = 0.0
        filter_errors_after = []
        for i, f in enumerate(filter_list):
            error = np.sum(np.abs(hists_syn_array[i] - hists_ori_array[i]) * weight) 
            filter_errors_after.append(error)
        max_err = np.max(filter_errors_after)
        sum_err = np.sum(filter_errors_after)
        iterations.append(round + 1)
        max_errors.append(max_err)
        sum_errors.append(sum_err)
        
        print(f"Max error after Gibbs: {max_err:.4f}")
        print(f"Sum error after Gibbs: {sum_err:.4f}")
        
        
        # 保存中间结果
        synthetic = img_syn / 7 * 255
        if save_img:
            cv2.imwrite(f"results/{img_name.split('.')[0]}/synthesized_{round}.jpg", synthetic)
        
        round += 1
        plot_errors(iterations, max_errors, sum_errors, img_name, save_img)

    
    
    # ==========================================
    # Step 8: 输出最终结果
    # ==========================================
    print(f"\n=== Synthesis Complete ===")
    print(f"Total rounds: {round}")
    print(f"Total selected filters: {len(filter_list)}")
    print(f"Final max error: {max_error:.4f}")
    
    # 保存最终图像
    final_synthetic = img_syn / 7 * 255
    
    if save_img:
        cv2.imwrite(f"results/{img_name.split('.')[0]}/final_synthesized.jpg", final_synthetic)
    
    return img_syn
    while max_error > threshold: # Not empty

        # TODO


        # save the synthesized image
        synthetic = img_syn / 7 * 255
        if save_img:
            cv2.imwrite(f"results/{img_name.split('.')[0]}/synthesized_{round}.jpg", synthetic)
        round += 1
    return img_syn  # return for testing
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Julesz ')
    parser.add_argument('--img_size', type=int, default=64, help='The size of the image.')
    parser.add_argument('--img_name', type=str, default='fur_obs.jpg', help='The name of the image.')
    args = parser.parse_args()

    julesz(img_size=args.img_size, img_name=args.img_name, save_img=True)
    #test_image = np.random.rand(7, 7)
    #filters = get_filters()
    


    #print(f"Number of filters: {len(filters)}")
    #for i, f in enumerate(filters[:]):  # 测试前3个

        #response1 = conv(test_image, f)
        #print(response1)
        #print(f"Filter {i}: input {test_image.shape}, ")
    #julesz()
