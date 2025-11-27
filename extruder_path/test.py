import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Use simple English labels to avoid font issues
plt.rcParams['font.family'] = 'DejaVu Sans'


# Read curve file starting from a specific integer
def read_curve(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find the first integer between 200 and 800
        start_index = None
        target_number = None
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    first_num = float(parts[0])
                    # Check if it's an integer and in the range 200-800
                    if first_num.is_integer() and 200 < int(first_num) < 800:
                        start_index = i
                        target_number = int(first_num)
                        break
                except ValueError:
                    continue

        if start_index is None:
            print(f"Warning: No integer in range 200-800 found in {file_path}, skipping file.")
            return np.array([])  # Return empty array if no valid starting point found
        else:
            data_lines = lines[start_index:]
            print(f"Found starting integer {target_number} in {file_path}, reading from line {start_index + 1}.")

        # Parse data lines
        data = []
        for line in data_lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    data.append([float(parts[0]), float(parts[2])])
                except ValueError:
                    continue

        curve = np.array(data)
        return curve

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return np.array([])
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return np.array([])


# DTW function with progress bar
def dtw_with_progress(s1, s2):
    n, m = len(s1), len(s2)

    if n == 0 or m == 0:
        return np.inf

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    total_steps = n * m
    disable_tqdm = False
    try:
        from sys import stdout
        if not stdout.isatty():
            disable_tqdm = True
    except:
        pass

    progress_bar = tqdm(total=total_steps, desc="DTW Calculation", disable=disable_tqdm)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt((s1[i - 1][0] - s2[j - 1][0]) ** 2 + (s1[i - 1][1] - s2[j - 1][1]) ** 2)
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
            progress_bar.update(1)

    progress_bar.close()
    return dtw_matrix[n, m]


# Visualize DTW results
def visualize_dtw_results(comparison_results, test_curve_filename, save_path=None):
    valid_results = [r for r in comparison_results if r['dtw_distance'] != np.inf]

    if not valid_results:
        print("No valid DTW distance results for visualization.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'DTW Distance Analysis - Reference: {test_curve_filename}', fontsize=16, y=0.98)

    filenames = [r['file'] for r in valid_results]
    distances = [r['dtw_distance'] for r in valid_results]

    # 1. Bar chart of all DTW distances
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(filenames)), distances, color='skyblue', alpha=0.7)
    ax1.set_title('DTW Distances for All Files', fontsize=14)
    ax1.set_xlabel('File Index', fontsize=12)
    ax1.set_ylabel('DTW Distance', fontsize=12)
    ax1.grid(True, alpha=0.3)

    min_idx = np.argmin(distances)
    ax1.bar(min_idx, distances[min_idx], color='red', alpha=0.8)
    ax1.text(min_idx, distances[min_idx] + max(distances) * 0.02,
             f'Most Similar\n{filenames[min_idx][:15]}...',
             ha='center', va='bottom', fontsize=8, color='red')

    # 2. Distance distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(distances, bins=min(20, len(distances)), color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_title('DTW Distance Distribution', fontsize=14)
    ax2.set_xlabel('DTW Distance', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.2f}')
    ax2.axvline(np.median(distances), color='blue', linestyle='--', label=f'Median: {np.median(distances):.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Top 10 most similar files
    ax3 = axes[1, 0]
    top_10 = sorted(valid_results, key=lambda x: x['dtw_distance'])[:10]
    top_10_names = [r['file'][:20] + '...' if len(r['file']) > 20 else r['file'] for r in top_10]
    top_10_distances = [r['dtw_distance'] for r in top_10]

    y_pos = np.arange(len(top_10_names))
    bars = ax3.barh(y_pos, top_10_distances, color='orange', alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_10_names, fontsize=9)
    ax3.set_xlabel('DTW Distance', fontsize=12)
    ax3.set_title('Top 10 Most Similar Files', fontsize=14)
    ax3.grid(True, alpha=0.3)

    for i, (bar, dist) in enumerate(zip(bars, top_10_distances)):
        ax3.text(bar.get_width() + max(top_10_distances) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{dist:.2f}', ha='left', va='center', fontsize=8)

    # 4. Statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
Statistics:

Reference curve: {test_curve_filename}

Total files compared: {len(comparison_results)}
Valid comparisons: {len(valid_results)}

DTW Distance Statistics:
• Min: {min(distances):.4f}
• Max: {max(distances):.4f}
• Mean: {np.mean(distances):.4f}
• Median: {np.median(distances):.4f}
• Std Dev: {np.std(distances):.4f}

Most similar file:
{filenames[min_idx]}
DTW Distance: {distances[min_idx]:.4f}
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


# Visualize curve comparison
def visualize_curve_comparison(test_curve, most_similar_curve, test_filename, similar_filename, save_path=None):
    plt.figure(figsize=(12, 8))

    plt.plot(test_curve[:, 0], test_curve[:, 1], 'b-', linewidth=2,
             label=f'Reference curve: {test_filename}', alpha=0.8)
    plt.plot(most_similar_curve[:, 0], most_similar_curve[:, 1], 'r--', linewidth=2,
             label=f'Most similar curve: {similar_filename}', alpha=0.8)

    plt.xlabel('X coordinate', fontsize=12)
    plt.ylabel('Z coordinate', fontsize=12)
    plt.title('Reference vs Most Similar Curve Comparison', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    stats_text = f"""
Curve Information:
Reference curve points: {len(test_curve)}
Similar curve points: {len(most_similar_curve)}
    """
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curve comparison saved to: {save_path}")

    plt.show()


# Main execution logic
def compare_all_curves():
    test_curve_filename = "00000.txt"
    curves_directory = r"/home/liu/桌面/amorsyn/data/carbon-fiber/"

    visualization_save_path = os.path.join(curves_directory, "dtw_analysis_results.png")
    curve_comparison_save_path = os.path.join(curves_directory, "curve_comparison.png")

    if not os.path.isdir(curves_directory):
        print(f"Error: Directory '{curves_directory}' does not exist.")
        return

    test_curve_full_path = os.path.join(curves_directory, test_curve_filename)

    print(f"Reading reference curve: {test_curve_filename} ...")
    test_curve = read_curve(test_curve_full_path)
    if test_curve.size == 0:
        print(f"Error: Reference curve '{test_curve_filename}' is empty or cannot be read.")
        return
    print(f"Reference curve '{test_curve_filename}' loaded. Points: {len(test_curve)}")

    comparison_results = []

    print(f"\nStarting comparison of other .txt curves with '{test_curve_filename}'...")

    all_files_in_directory = [f for f in os.listdir(curves_directory)
                              if f.endswith(".txt") and os.path.isfile(os.path.join(curves_directory, f))]

    if not all_files_in_directory:
        print(f"No .txt files found in directory '{curves_directory}'.")
        return

    for filename in tqdm(all_files_in_directory, desc="File comparison progress"):
        if filename == test_curve_filename:
            continue

        current_file_path = os.path.join(curves_directory, filename)
        current_curve = read_curve(current_file_path)
        if current_curve.size == 0:
            print(f"\nWarning: File '{filename}' skipped (no valid starting integer 200-800 or empty file).")
            comparison_results.append(
                {'file': filename, 'dtw_distance': np.inf, 'error': 'No valid starting integer or empty file'})
            continue

        distance = dtw_with_progress(test_curve, current_curve)
        comparison_results.append({'file': filename, 'dtw_distance': distance})

    if not comparison_results:
        print(f"\nNo other curves found for comparison in directory '{curves_directory}'.")
    else:
        print("\n\n--- All curve comparison results ---")
        comparison_results_sorted = sorted(comparison_results, key=lambda x: x.get('dtw_distance', np.inf))

        for result in comparison_results_sorted:
            dist_str = f"{result['dtw_distance']:.4f}" if result['dtw_distance'] != np.inf else "Infinity"
            if 'error' in result:
                print(f"File: {result['file']:<40} DTW Distance: {dist_str} (Note: {result['error']})")
            else:
                print(f"File: {result['file']:<40} DTW Distance: {dist_str}")

        most_similar = None
        for res in comparison_results_sorted:
            if res['dtw_distance'] != np.inf:
                most_similar = res
                break

        if most_similar:
            print(
                f"\nMost similar curve to '{test_curve_filename}' is '{most_similar['file']}' with DTW distance: {most_similar['dtw_distance']:.4f}")

            print("\nGenerating visualization charts...")

            visualize_dtw_results(comparison_results, test_curve_filename, visualization_save_path)

            most_similar_curve_path = os.path.join(curves_directory, most_similar['file'])
            most_similar_curve = read_curve(most_similar_curve_path)
            if most_similar_curve.size > 0:
                visualize_curve_comparison(test_curve, most_similar_curve,
                                           test_curve_filename, most_similar['file'],
                                           curve_comparison_save_path)

        else:
            print(
                f"\nNo valid similar curves found for '{test_curve_filename}' (all comparison results are infinity or failed to read).")


if __name__ == "__main__":
    compare_all_curves()