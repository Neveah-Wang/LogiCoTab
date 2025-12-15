import os
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, entropy
import numpy as np

# Function to perform t-SNE, plot KDE, and compute Wasserstein and JS distances
def perform_tsne_and_plot(original_data_file, generated_data_file, save_folder):
    # Read original and generated data
    original_data = pd.read_csv(original_data_file, header=None)
    generated_data = pd.read_csv(generated_data_file, header=None)

    # Combine original and generated data for t-SNE
    all_samples = np.vstack((original_data.values, generated_data.values))
    labels = np.array([0] * len(original_data) + [1] * len(generated_data))

    # Dynamically adjust perplexity parameter
    perplexity = min(30, len(all_samples) - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(all_samples)

    # Separate the t-SNE results back into original and generated
    original_data_tsne = tsne_results[labels == 0]
    generated_data_tsne = tsne_results[labels == 1]

    # Plot t-SNE scatter plot with square aspect ratio
    plt.figure(figsize=(8, 8))
    plt.scatter(original_data_tsne[:, 0], original_data_tsne[:, 1], alpha=0.5, label='Real Data', s=188)
    plt.scatter(generated_data_tsne[:, 0], generated_data_tsne[:, 1], alpha=0.5, label='Generated Data', s=188)
    # 增加图例的字体大小
    plt.legend(fontsize=58)
    plt.title('t-SNE Visualization', fontsize=20)
    # 设置刻度值的大小
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend()

    # Save t-SNE plot
    tsne_plot_filename = os.path.join(save_folder, 'tsne_plot.png')
    plt.tight_layout()
    plt.savefig(tsne_plot_filename)
    plt.close()

    # Plot KDE for each t-SNE component
    plt.figure(figsize=(16, 8))

    # KDE for t-SNE component 1
    plt.subplot(1, 2, 1)
    sns.kdeplot(original_data_tsne[:, 0], label='Original Data', fill=True)
    sns.kdeplot(generated_data_tsne[:, 0], label='Generated Data', fill=True)
    plt.title('Kernel Density Estimation', fontsize=18)
    plt.xlabel('Feature 1', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    # 设置刻度值的大小
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend()

    # KDE for t-SNE component 2
    plt.subplot(1, 2, 2)
    sns.kdeplot(original_data_tsne[:, 1], label='Original Data', fill=True)
    sns.kdeplot(generated_data_tsne[:, 1], label='Generated Data', fill=True)
    plt.title('Kernel Density Estimation', fontsize=18)
    plt.xlabel('Feature 2', fontsize=16)
    plt.ylabel('Density', fontsize=16)

    # 设置刻度值的大小
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend()

    # Save KDE plots
    kde_plot_filename = os.path.join(save_folder, 'kde_tsne_plot.png')
    plt.tight_layout()
    plt.savefig(kde_plot_filename)
    plt.close()

    # Plot KDE for each feature
    num_features = original_data.shape[1]
    plt.figure(figsize=(8 * num_features, 8))

    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        sns.kdeplot(original_data.iloc[:, i], label='Original Data', fill=True)
        sns.kdeplot(generated_data.iloc[:, i], label='Generated Data', fill=True)
        plt.title(f'Feature {i + 1} KDE')
        plt.xlabel(f'Feature {i + 1} Value')
        plt.ylabel('Density')
        plt.legend()

    # Save plot
    plot_filename = os.path.join(save_folder, 'kde_plots.png')
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    # Compute Wasserstein distance using original data
    wasserstein_dist = wasserstein_distance(original_data.values.flatten(), generated_data.values.flatten())

    # Compute JS distance using original data
    # Compute histograms with the same bins
    hist_original, _ = np.histogram(original_data.values.flatten(), bins=10, density=True)
    hist_generated, _ = np.histogram(generated_data.values.flatten(), bins=10, density=True)

    # Compute JS distance
    js_dist = np.sqrt(0.5 * (entropy(hist_original, (hist_original + hist_generated) / 2) +
                             entropy(hist_generated, (hist_original + hist_generated) / 2)))

    # Compute L2 distance if dimensions match
    if original_data.shape[0] == generated_data.shape[0]:
        l2_dist = np.linalg.norm(original_data.values - generated_data.values, ord=2)
    else:
        l2_dist = None

    # Save distances to files
    wasserstein_filename = os.path.join(save_folder, 'wasserstein_distance.txt')
    with open(wasserstein_filename, 'w') as f:
        f.write(f'Wasserstein Distance: {wasserstein_dist}')

    js_filename = os.path.join(save_folder, 'js_distance.txt')
    with open(js_filename, 'w') as f:
        f.write(f'JS Distance: {js_dist}')

    l2_filename = os.path.join(save_folder, 'l2_distance.txt')
    with open(l2_filename, 'w') as f:
        f.write(f'L2 Distance: {l2_dist}')

    # Calculate correlation matrices and their absolute differences
    real_corr = original_data.corr()
    generated_corr = generated_data.corr()
    diff_corr = np.abs(real_corr - generated_corr)

    # Replace NaN values with 0
    diff_corr = diff_corr.fillna(0)

    # Plot heatmap of the absolute difference
    plt.figure(figsize=(10, 8))
    # sns.heatmap(diff_corr, cmap='Reds', cbar=False, vmin=0, vmax=2, annot=True, fmt=".2f")
    sns.heatmap(diff_corr, cmap='Reds', cbar=False, vmin=0, vmax=2, annot=False)
    plt.title('Absolute difference between correlation matrices', fontsize=20)
    plt.xlabel('Features', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    # 设置刻度值的大小
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Save plot
    heatmap_filename = os.path.join(save_folder, 'correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_filename)
    plt.close()

    # Calculate and print the sum of all values in the difference correlation matrix
    total_sum = diff_corr.values.sum()
    print(f'Sum of all values in the correlation heatmap: {total_sum}')

    # Save the sum to a file
    sum_filename = os.path.join(save_folder, 'sum_of_correlation_heatmap.txt')
    with open(sum_filename, 'w') as f:
        f.write(f'Sum of all values in the correlation heatmap: {total_sum}')

for i in range(3, 4):
    # Define paths
    original_data_folder = f'../processed_data/imb_{i}/'
    generated_data_folder = f'../processed_data/imb_{i}/'
    result_folder = f'../../result/GAN/'

    # Iterate through datasets
    for dataset_folder in os.listdir(original_data_folder):
        # Ensure result folder exists, create if not
        os.makedirs(result_folder, exist_ok=True)
        original_data_path = os.path.join(original_data_folder, dataset_folder, 'minority.csv')
        generated_data_path = os.path.join(generated_data_folder, dataset_folder, 'generated.csv')

        # Create result folder for each dataset
        dataset_result_folder = os.path.join(result_folder, dataset_folder)
        os.makedirs(dataset_result_folder, exist_ok=True)

        # Perform t-SNE, plot KDE, and compute distances
        perform_tsne_and_plot(original_data_path, generated_data_path, dataset_result_folder)
    print(f'imb_{i}完成测试')
