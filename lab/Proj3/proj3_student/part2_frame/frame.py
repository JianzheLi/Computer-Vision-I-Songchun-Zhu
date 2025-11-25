'''
This is the main file of Part 2: FRAME Model
'''

import numpy as np
from filters_frame import get_filters
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import os
# 尝试导入Cython版本
try:
    from gibbs_frame import gibbs_sample_frame_cy
    USE_CYTHON = True
    print("Using Cython-optimized Gibbs sampler")
except ImportError:
    USE_CYTHON = False
    print("Cython version not found, using pure Python version")
#USE_CYTHON = False
from gibbs import gibbs_sample



def plot_lambda_evolution(lambda_history, iteration_numbers, save_dir, num_filters):
    """
    绘制 lambda 参数随迭代次数的变化
    
    参数:
    - lambda_history: list of arrays, 每个元素是 (num_filters, num_bins) 的数组
    - iteration_numbers: list, 迭代次数
    - save_dir: str, 保存目录
    - num_filters: int, 滤波器数量
    """
    lambda_history = np.array(lambda_history)  # shape: (iterations, num_filters, num_bins)
    
    # 方法1: 绘制每个滤波器的 lambda 范数（所有 bins 的平方和的平方根）
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for filter_idx in range(num_filters):
        # 计算每次迭代中该滤波器所有 bins 的 L2 范数
        lambda_norms = np.linalg.norm(lambda_history[:, filter_idx, :], axis=1)
        ax.plot(iteration_numbers, lambda_norms, label=f'Filter {filter_idx+1}', alpha=0.7)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('||λ|| (L2 Norm)', fontsize=12)
    ax.set_title('Evolution of Lambda Parameters Over Iterations', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lambda_evolution_all_filters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 方法2: 绘制选定几个滤波器的详细变化（例如前4个）
    num_display = min(4, num_filters)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i in range(num_display):
        ax = axes[i]
        # 绘制该滤波器每个 bin 的 lambda 值
        for bin_idx in range(lambda_history.shape[2]):
            ax.plot(iteration_numbers, lambda_history[:, i, bin_idx], 
                   label=f'Bin {bin_idx+1}', alpha=0.6)
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('λ value', fontsize=10)
        ax.set_title(f'Filter {i+1}: Lambda Evolution by Bin', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lambda_evolution_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 方法3: 热图显示所有 lambda 的演化
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 将 lambda 展平为 (iterations, num_filters * num_bins)
    lambda_flat = lambda_history.reshape(len(iteration_numbers), -1)
    
    im = ax.imshow(lambda_flat.T, aspect='auto', cmap='RdBu_r', 
                   interpolation='nearest', extent=[iteration_numbers[0], iteration_numbers[-1], 
                                                   0, lambda_flat.shape[1]])
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Lambda Index (Filter × Bin)', fontsize=12)
    ax.set_title('Heatmap of All Lambda Parameters Over Iterations', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('λ value', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lambda_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nLambda evolution plots saved to {save_dir}/")

def conv(image, filter):
    ''' 
    Computes the filter response on an image using circular padding.
    '''
    filtered_image = convolve2d(image, filter, mode='same', boundary='wrap')
    return filtered_image

def get_histogram(filtered_image, bins_num, max_response, min_response, img_H, img_W):
    ''' 
    Computes the normalized filter response histogram.
    '''
    if max_response <= min_response:
        max_response = min_response + 1e-6
    
    edges = np.linspace(min_response, max_response, bins_num + 1)
    hist, _ = np.histogram(filtered_image.ravel(), bins=edges)
    hist = hist.astype(np.float64) / (img_H * img_W)
    return hist

def frame(img_size=64, img_name="fur_obs.jpg", save_img=True):
    '''
    FRAME model implementation
    '''
    max_intensity = 255
    
    # Get all filters
    F_list = get_filters()
    F_list = [f.astype(np.float32) for f in F_list]
    num_filters = len(F_list)
    
    print(f"Total number of filters: {num_filters}")
    
    # Image size
    img_H = img_W = img_size
    
    # Read original image
    img_ori = cv2.resize(cv2.imread(f'images/{img_name}', cv2.IMREAD_GRAYSCALE), (img_H, img_W))
    img_ori = (img_ori).astype(np.float32)
    img_ori = img_ori * 7 // max_intensity

    if save_img:
        save_dir = f"results/{img_name.split('.')[0]}"
        os.makedirs(save_dir, exist_ok=True)  
    if save_img:
        cv2.imwrite(f"results/{img_name.split('.')[0]}/original.jpg", (img_ori / 7 * 255))
    
    # Initialize synthetic image
    img_syn = np.random.randint(0, 8, img_ori.shape).astype(np.float32)
    
    # Parameters
    num_bins = 15
    threshold = 0.1
    max_iterations = 50
    gibbs_sweeps = 50
    initial_temperature = 1.0
    learning_rate = 0.01
    lambda_history = []  # 存储每次迭代的 lambda
    iteration_numbers = []  # 存储迭代次数
    
    # Precompute original image responses and bounds
    print("Computing filter responses on original image...")
    ori_responses = []
    bounds = []
    ori_hists = []
    
    for f in F_list:
        resp = conv(img_ori, f)
        ori_responses.append(resp)
        max_resp = np.max(resp)
        min_resp = np.min(resp)
        bounds.append([max_resp, min_resp])  # Note: [max, min] order
        hist = get_histogram(resp, num_bins, max_resp, min_resp, img_H, img_W)
        ori_hists.append(hist)
    
    ori_hists = np.array(ori_hists, dtype=np.float32)
    bounds = np.array(bounds, dtype=np.float32)
    
    # Initialize lambda (Lagrange multipliers)
    lambdas = np.zeros((num_filters, num_bins), dtype=np.float32)
    
    print("\n---- FRAME Model Training ----")
    
    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        
        # Gibbs sampling to update image
        if USE_CYTHON:
            img_syn = gibbs_sample_frame_cy(
                img_syn, F_list, lambdas, bounds, num_bins,
                gibbs_sweeps, initial_temperature, num_threads=8)
        else:
            # Fallback to Python version (you'd need to implement this)
            try :
                img_syn = gibbs_sample_frame(
                img_syn, F_list, lambdas, bounds, num_bins,
                gibbs_sweeps, initial_temperature)
                
            except:

                raise NotImplementedError("Pure Python version not available, please compile Cython version")
        
        # Compute synthetic image histograms
        syn_hists = []
        for idx, f in enumerate(F_list):
            syn_resp = conv(img_syn, f)
            max_resp, min_resp = bounds[idx]
            hist_syn = get_histogram(syn_resp, num_bins, max_resp, min_resp, img_H, img_W)
            syn_hists.append(hist_syn)
        syn_hists = np.array(syn_hists, dtype=np.float32)
        
        # Gradient ascent on log p(x; Lambda)
        # d(lambda_n)/dt = E_p[phi_n(x)] - mu_n = hist_syn - hist_ori
        gradient = syn_hists - ori_hists
        lambdas += learning_rate * gradient
        lambda_history.append(lambdas.copy())  
        iteration_numbers.append(iteration + 1)
        # Compute error
        errors = np.abs(syn_hists - ori_hists)
        max_error = errors.sum(axis=1).max()
        mean_error = errors.sum(axis=1).mean()
        
        print(f'Iteration {iteration + 1}: mean_error = {mean_error:.6f}, max_error = {max_error:.6f}')
        
        # Save intermediate results
        if save_img:
            synthetic = img_syn / 7 * 255
            cv2.imwrite(f"results/{img_name.split('.')[0]}/synthesized_{iteration+1}.jpg", synthetic)
            print("saved")

        if save_img:
            plot_lambda_evolution(lambda_history, iteration_numbers, save_dir, num_filters)
    # Save final result
        # Check convergence
        if max_error < threshold:
            print(f"Converged at iteration {iteration + 1}")
            break
        
        
    final_synthetic = img_syn / 7 * 255

    if save_img:
        
        cv2.imwrite(f"results/{img_name.split('.')[0]}/final_synthesized.jpg", final_synthetic)

    
    print(f"\n=== FRAME Training Complete ===")
    print(f"Total iterations: {iteration + 1}")
    print(f"Final max error: {max_error:.6f}")
    
    return img_syn

if __name__ == '__main__':
    frame()


