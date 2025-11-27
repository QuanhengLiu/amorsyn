import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool
from ruamel.yaml import YAML
import yaml as std_yaml  # 备用YAML处理器
import shutil
import time
import traceback

# Set matplotlib font
plt.rcParams['font.family'] = 'DejaVu Sans'


@dataclass
class ParameterConfig:
    """Parameter configuration data class"""
    name: str
    min_value: float
    max_value: float
    current_range: Tuple[float, float]
    step_size: float
    file_prefix: str
    description: str

    def __post_init__(self):
        if self.current_range is None:
            self.current_range = (self.min_value, self.max_value)


@dataclass
class OptimizationResult:
    """Optimization result data class"""
    iteration: int
    best_params: Dict[str, float]
    best_distance: float
    best_file: str
    convergence_history: List[float]
    param_history: List[Dict[str, float]]
    total_evaluations: int
    computation_time: float


class IterativeParameterOptimizer:
    """Iterative parameter optimizer"""

    def __init__(self,
                 base_config_path: str,
                 target_curve_file: str,
                 work_directory: str,
                 exec_file_path: str):
        """
        Initialize optimizer

        Args:
            base_config_path: Base YAML configuration file path
            target_curve_file: Target curve file name
            work_directory: Working directory
            exec_file_path: 3D printing simulation program path
        """
        self.base_config_path = Path(base_config_path)
        self.target_curve_file = target_curve_file
        self.work_dir = Path(work_directory)
        self.exec_file_path = exec_file_path

        # Create working directory structure
        self.iterations_dir = self.work_dir / "iterations"
        self.results_dir = self.work_dir / "results"
        self.iterations_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize YAML processor
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.width = 4096

        # Load base configuration with error handling
        try:
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                self.base_config = self.yaml.load(f)
        except Exception as e:
            print(f"Warning: Failed to load with ruamel.yaml, using standard yaml: {e}")
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                self.base_config = std_yaml.safe_load(f)

        # Optimization history
        self.optimization_history = []
        self.current_iteration = 0

        # Parameter configuration
        self.parameters = self._initialize_parameters()

        print(f"Optimizer initialization completed")
        print(f"Working directory: {self.work_dir}")
        print(f"Target curve: {self.target_curve_file}")
        print(f"Number of parameters: {len(self.parameters)}")

    def _initialize_parameters(self) -> Dict[str, ParameterConfig]:
        """Initialize parameter configuration"""
        return {
            'ropeSpringStiffness': ParameterConfig(
                name='ropeSpringStiffness',
                min_value=10.0,
                max_value=25.0,
                current_range=(15.0, 20.0),
                step_size=0.1,
                file_prefix='stiff',
                description='Rope spring stiffness'
            ),
            'printerFrictionCoefficient': ParameterConfig(
                name='printerFrictionCoefficient',
                min_value=0.8,
                max_value=2.0,
                current_range=(1.2, 1.6),
                step_size=0.01,
                file_prefix='fric',
                description='Printer friction coefficient'
            ),
            'printerMass': ParameterConfig(
                name='printerMass',
                min_value=600,
                max_value=1400,
                current_range=(800, 1200),
                step_size=10,
                file_prefix='mass',
                description='Printer mass'
            ),
            'ropeIndividualMass': ParameterConfig(
                name='ropeIndividualMass',
                min_value=0.5,
                max_value=1.5,
                current_range=(0.9, 1.1),
                step_size=0.02,
                file_prefix='indmass',
                description='Rope individual mass'
            ),
            'groundFrictionCoefficient': ParameterConfig(
                name='groundFrictionCoefficient',
                min_value=50.0,
                max_value=150.0,
                current_range=(80.0, 120.0),
                step_size=2.0,
                file_prefix='ground',
                description='Ground friction coefficient'
            )
        }

    def generate_parameter_combinations(self,
                                        strategy: str = "adaptive_grid",
                                        n_samples: int = 50) -> List[Dict[str, float]]:
        """
        Generate parameter combinations

        Args:
            strategy: Sampling strategy ('grid', 'random', 'adaptive_grid', 'latin_hypercube')
            n_samples: Number of samples

        Returns:
            List of parameter combinations
        """
        if strategy == "grid":
            return self._generate_grid_samples(n_samples)
        elif strategy == "random":
            return self._generate_random_samples(n_samples)
        elif strategy == "adaptive_grid":
            return self._generate_adaptive_grid_samples(n_samples)
        elif strategy == "latin_hypercube":
            return self._generate_lhs_samples(n_samples)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def _generate_grid_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate grid sampling"""
        n_per_param = max(2, int(n_samples ** (1 / len(self.parameters))))
        combinations = []

        param_values = {}
        for param_name, config in self.parameters.items():
            start, end = config.current_range
            param_values[param_name] = np.linspace(start, end, n_per_param)

        # Generate Cartesian product
        import itertools
        param_names = list(param_values.keys())
        for combination in itertools.product(*[param_values[name] for name in param_names]):
            param_dict = {}
            for name, value in zip(param_names, combination):
                # 限制精度，避免极高精度浮点数
                param_dict[name] = round(float(value), 6)
            combinations.append(param_dict)

        return combinations[:n_samples]

    def _generate_random_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate random sampling"""
        combinations = []
        for _ in range(n_samples):
            param_dict = {}
            for param_name, config in self.parameters.items():
                start, end = config.current_range
                value = np.random.uniform(start, end)
                # 限制精度，避免极高精度浮点数
                param_dict[param_name] = round(float(value), 6)
            combinations.append(param_dict)
        return combinations

    def _generate_adaptive_grid_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate adaptive grid sampling (dense sampling around historical optimal points)"""
        combinations = []

        # Base random sampling
        base_samples = n_samples // 2
        combinations.extend(self._generate_random_samples(base_samples))

        # If there are historical optimal points, sample densely around them
        if self.optimization_history:
            best_result = min(self.optimization_history, key=lambda x: x.best_distance)
            best_params = best_result.best_params

            remaining_samples = n_samples - base_samples
            for _ in range(remaining_samples):
                param_dict = {}
                for param_name, config in self.parameters.items():
                    if param_name in best_params:
                        # Sample around optimal point
                        center = best_params[param_name]
                        start, end = config.current_range
                        range_size = end - start
                        noise_range = range_size * 0.1  # Perturbation within 10% range

                        value = np.random.normal(center, noise_range / 3)
                        value = np.clip(value, start, end)
                        param_dict[param_name] = round(float(value), 6)
                    else:
                        start, end = config.current_range
                        value = np.random.uniform(start, end)
                        param_dict[param_name] = round(float(value), 6)
                combinations.append(param_dict)

        return combinations

    def _generate_lhs_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate Latin Hypercube Sampling"""
        try:
            from scipy.stats import qmc

            # Create LHS sampler
            sampler = qmc.LatinHypercube(d=len(self.parameters))
            samples = sampler.random(n=n_samples)

            combinations = []
            param_names = list(self.parameters.keys())

            for sample in samples:
                param_dict = {}
                for i, param_name in enumerate(param_names):
                    config = self.parameters[param_name]
                    start, end = config.current_range
                    # Map [0,1] interval sample values to parameter range
                    value = start + sample[i] * (end - start)
                    param_dict[param_name] = round(float(value), 6)
                combinations.append(param_dict)

            return combinations
        except ImportError:
            print("scipy not available, using random sampling instead of LHS")
            return self._generate_random_samples(n_samples)

    def generate_config_files(self, param_combinations: List[Dict[str, float]],
                              iteration: int) -> Tuple[List[Tuple[str, str]], str]:
        """
        Generate configuration files

        Args:
            param_combinations: List of parameter combinations
            iteration: Current iteration number

        Returns:
            (List of configuration file paths, list file path)
        """
        iter_dir = self.iterations_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(exist_ok=True)

        generated_files = []

        print(f"Generating configuration files for iteration {iteration}...")

        for i, params in enumerate(tqdm(param_combinations, desc="Generating config files")):
            try:
                # Generate filename - 限制浮点数精度
                param_str = "_".join([f"{name}_{value:.6f}" for name, value in params.items()])
                yaml_file = iter_dir / f"config_{i:04d}_{param_str}.yaml"
                txt_file = yaml_file.with_suffix('.txt')

                # Copy base configuration and modify parameters
                config = dict(self.base_config)
                for param_name, value in params.items():
                    # 确保参数值是标准的Python float类型
                    config[param_name] = float(value)

                # Write configuration file with error handling
                success = False
                try:
                    with open(yaml_file, 'w', encoding='utf-8') as f:
                        self.yaml.dump(config, f)
                    success = True
                except Exception as yaml_error:
                    print(f"ruamel.yaml writing error for config {i}: {yaml_error}")
                    # 尝试使用标准库的yaml
                    try:
                        with open(yaml_file, 'w', encoding='utf-8') as f:
                            std_yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                        success = True
                    except Exception as std_yaml_error:
                        print(f"Standard yaml writing error for config {i}: {std_yaml_error}")

                if success:
                    generated_files.append((str(yaml_file), str(txt_file)))
                else:
                    print(f"Failed to generate config file {i}")

            except Exception as e:
                print(f"Error generating config file {i}: {e}")
                continue

        # Generate list file with error handling
        list_file = iter_dir / "list"
        try:
            with open(list_file, 'w', encoding='utf-8') as f:
                for yaml_path, txt_path in generated_files:
                    f.write(f"{yaml_path},{txt_path}\n")
        except Exception as e:
            print(f"Error writing list file: {e}")
            raise

        print(f"Generated {len(generated_files)} configuration files")
        return generated_files, str(list_file)

    def run_simulations(self, list_file: str, pool_size: int = 44) -> int:
        """
        Run simulations

        Args:
            list_file: Task list file
            pool_size: Process pool size

        Returns:
            Number of successfully executed tasks
        """
        print("Starting simulation execution...")

        tasks = []
        try:
            with open(list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    yaml_path, txt_path = line.split(',')

                    # Check if output file already exists
                    if os.path.exists(txt_path):
                        continue

                    cmd_line = f'{self.exec_file_path} {yaml_path} {txt_path} VIS_OFF'
                    tasks.append(cmd_line)
        except Exception as e:
            print(f"Error reading list file: {e}")
            return 0

        if not tasks:
            print("All simulations already completed")
            return 0

        print(f"Need to execute {len(tasks)} simulation tasks")

        # Execute simulations
        try:
            with Pool(pool_size) as p:
                results = list(tqdm(p.imap_unordered(os.system, tasks),
                                    total=len(tasks), desc="Executing simulations"))

            successful_tasks = sum(1 for r in results if r == 0)
            print(f"Successfully completed {successful_tasks}/{len(tasks)} simulations")

            return successful_tasks
        except Exception as e:
            print(f"Error in simulation execution: {e}")
            return 0

    def read_curve(self, file_path: str) -> np.ndarray:
        """Read curve data with retry mechanism"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 确保文件存在且可读
                if not os.path.exists(file_path):
                    return np.array([])

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Find starting integer in 200-800 range
                start_index = None
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            first_num = float(parts[0])
                            if first_num.is_integer() and 200 < int(first_num) < 800:
                                start_index = i
                                break
                        except ValueError:
                            continue

                if start_index is None:
                    return np.array([])

                # Parse data
                data = []
                for line in lines[start_index:]:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            data.append([float(parts[0]), float(parts[2])])
                        except ValueError:
                            continue

                return np.array(data)

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed to read {file_path}: {e}. Retrying...")
                    time.sleep(0.1)  # 短暂等待后重试
                else:
                    print(f"Failed to read file {file_path} after {max_retries} attempts: {e}")
                    return np.array([])

    def calculate_dtw_distance(self, curve1: np.ndarray, curve2: np.ndarray) -> float:
        """Calculate DTW distance"""
        n, m = len(curve1), len(curve2)

        if n == 0 or m == 0:
            return np.inf

        # Use memory-efficient DTW algorithm for large datasets
        if n * m > 1e6:
            return self._dtw_memory_efficient(curve1, curve2)

        # Standard DTW
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.sqrt((curve1[i - 1][0] - curve2[j - 1][0]) ** 2 +
                               (curve1[i - 1][1] - curve2[j - 1][1]) ** 2)
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],  # insertion
                    dtw_matrix[i, j - 1],  # deletion
                    dtw_matrix[i - 1, j - 1]  # match
                )

        return dtw_matrix[n, m]

    def _dtw_memory_efficient(self, curve1: np.ndarray, curve2: np.ndarray) -> float:
        """Memory-efficient DTW algorithm"""
        n, m = len(curve1), len(curve2)

        prev_row = np.full(m + 1, np.inf)
        curr_row = np.full(m + 1, np.inf)
        prev_row[0] = 0

        for i in range(1, n + 1):
            curr_row[0] = np.inf
            for j in range(1, m + 1):
                cost = np.sqrt((curve1[i - 1][0] - curve2[j - 1][0]) ** 2 +
                               (curve1[i - 1][1] - curve2[j - 1][1]) ** 2)
                curr_row[j] = cost + min(prev_row[j], curr_row[j - 1], prev_row[j - 1])

            prev_row, curr_row = curr_row, prev_row

        return prev_row[m]

    def analyze_results(self, generated_files: List[Tuple[str, str]],
                        iteration: int) -> Tuple[Dict[str, float], float, str]:
        """
        Analyze results

        Args:
            generated_files: List of generated files
            iteration: Current iteration number

        Returns:
            (Optimal parameters, optimal distance, optimal file name)
        """
        print("Starting result analysis...")

        # Read target curve
        target_curve_path = self.work_dir / self.target_curve_file
        target_curve = self.read_curve(str(target_curve_path))

        if target_curve.size == 0:
            raise ValueError(f"Unable to read target curve: {target_curve_path}")

        results = []

        for yaml_path, txt_path in tqdm(generated_files, desc="Calculating DTW distances"):
            # Check if output file exists
            if not os.path.exists(txt_path):
                continue

            # Read simulation result curve
            sim_curve = self.read_curve(txt_path)
            if sim_curve.size == 0:
                continue

            # Calculate DTW distance
            distance = self.calculate_dtw_distance(target_curve, sim_curve)

            # Extract parameter values
            yaml_filename = Path(yaml_path).name
            params = self._extract_params_from_filename(yaml_filename)

            results.append({
                'params': params,
                'distance': distance,
                'yaml_file': yaml_path,
                'txt_file': txt_path
            })

        if not results:
            raise ValueError("No valid simulation results")

        # Find optimal result
        best_result = min(results, key=lambda x: x['distance'])

        # Save detailed results
        try:
            results_file = self.results_dir / f"iter_{iteration:03d}_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = []
                for r in results:
                    json_r = r.copy()
                    json_r['distance'] = float(json_r['distance']) if json_r['distance'] != np.inf else "infinity"
                    json_results.append(json_r)

                json.dump({
                    'iteration': iteration,
                    'total_results': len(results),
                    'best_result': {
                        'params': best_result['params'],
                        'distance': float(best_result['distance']),
                        'files': {
                            'yaml': best_result['yaml_file'],
                            'txt': best_result['txt_file']
                        }
                    },
                    'all_results': json_results
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save results file: {e}")

        print(f"Optimal result: DTW distance = {best_result['distance']:.6f}")
        print(f"Optimal parameters: {best_result['params']}")

        return best_result['params'], best_result['distance'], best_result['txt_file']

    def _extract_params_from_filename(self, filename: str) -> Dict[str, float]:
        """Extract parameter values from filename"""
        params = {}

        # Filename format: config_0000_param1_value1_param2_value2_....yaml
        parts = filename.replace('.yaml', '').split('_')

        i = 2  # Skip 'config' and sequence number
        while i < len(parts) - 1:
            param_name = parts[i]
            try:
                param_value = float(parts[i + 1])
                params[param_name] = param_value
                i += 2
            except (ValueError, IndexError):
                i += 1

        return params

    def update_parameter_ranges(self, best_params: Dict[str, float],
                                convergence_factor: float = 0.7):
        """
        Update parameter search ranges

        Args:
            best_params: Current optimal parameters
            convergence_factor: Convergence factor controlling search range reduction
        """
        print("Updating parameter search ranges...")

        for param_name, best_value in best_params.items():
            if param_name in self.parameters:
                config = self.parameters[param_name]
                current_start, current_end = config.current_range
                current_range_size = current_end - current_start

                # Calculate new search range
                new_range_size = current_range_size * convergence_factor
                new_start = best_value - new_range_size / 2
                new_end = best_value + new_range_size / 2

                # Ensure not exceeding absolute bounds
                new_start = max(new_start, config.min_value)
                new_end = min(new_end, config.max_value)

                # Update range
                config.current_range = (new_start, new_end)

                print(f"{config.description}: [{new_start:.4f}, {new_end:.4f}] "
                      f"(optimal value: {best_value:.4f})")

    def check_convergence(self, tolerance: float = 1e-4,
                          min_iterations: int = 3) -> bool:
        """
        Check convergence

        Args:
            tolerance: Convergence tolerance
            min_iterations: Minimum number of iterations

        Returns:
            Whether convergence is achieved
        """
        if len(self.optimization_history) < min_iterations:
            return False

        # Check improvement in recent iterations
        recent_distances = [result.best_distance for result in self.optimization_history[-3:]]

        # If change in optimal distance over last three iterations is small, consider converged
        if max(recent_distances) - min(recent_distances) < tolerance:
            return True

        return False

    def visualize_optimization_progress(self):
        """Visualize optimization progress"""
        if not self.optimization_history:
            print("No optimization history data")
            return

        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))
            fig.suptitle('Parameter Optimization Progress Analysis', fontsize=16)

            # 1. Convergence curve
            ax1 = axes[0, 0]
            iterations = [r.iteration for r in self.optimization_history]
            best_distances = [r.best_distance for r in self.optimization_history]

            ax1.plot(iterations, best_distances, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Best DTW Distance')
            ax1.set_title('Convergence Curve')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')

            # 2. Parameter evolution
            ax2 = axes[0, 1]
            param_names = list(self.parameters.keys())
            colors = plt.cm.Set3(np.linspace(0, 1, len(param_names)))

            for i, param_name in enumerate(param_names):
                param_values = []
                for result in self.optimization_history:
                    if param_name in result.best_params:
                        param_values.append(result.best_params[param_name])
                    else:
                        param_values.append(np.nan)

                if any(not np.isnan(v) for v in param_values):
                    ax2.plot(iterations, param_values, 'o-', color=colors[i],
                             label=param_name, linewidth=2, markersize=6)

            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Parameter Value')
            ax2.set_title('Parameter Evolution')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)

            # 3. Evaluation count statistics
            ax3 = axes[1, 0]
            evaluations = [r.total_evaluations for r in self.optimization_history]
            cumulative_evals = np.cumsum(evaluations)

            bars = ax3.bar(iterations, evaluations, alpha=0.7, color='lightblue',
                           label='Single evaluation')
            ax3_twin = ax3.twinx()
            ax3_twin.plot(iterations, cumulative_evals, 'ro-', linewidth=2,
                          label='Cumulative evaluations')

            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Single Evaluation Count', color='blue')
            ax3_twin.set_ylabel('Cumulative Evaluation Count', color='red')
            ax3.set_title('Evaluation Count Statistics')
            ax3.grid(True, alpha=0.3)

            # 4. Computation time statistics
            ax4 = axes[1, 1]
            comp_times = [r.computation_time for r in self.optimization_history]

            ax4.bar(iterations, comp_times, alpha=0.7, color='lightgreen')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Computation Time (seconds)')
            ax4.set_title('Computation Time Statistics')
            ax4.grid(True, alpha=0.3)

            # 5. 目标曲线与最优结果对比
            ax5 = axes[1, 2]
            try:
                # 读取目标曲线
                target_curve_path = self.work_dir / self.target_curve_file
                target_curve = self.read_curve(str(target_curve_path))

                # 获取最优结果
                best_result = min(self.optimization_history, key=lambda x: x.best_distance)
                best_curve = self.read_curve(best_result.best_file)

                if target_curve.size > 0:
                    ax5.plot(target_curve[:, 0], target_curve[:, 1], 'b-', linewidth=3,
                             label='Target Curve', alpha=0.8)

                if best_curve.size > 0:
                    ax5.plot(best_curve[:, 0], best_curve[:, 1], 'r--', linewidth=2,
                             label=f'Best Result (DTW: {best_result.best_distance:.4f})', alpha=0.8)

                ax5.set_xlabel('X')
                ax5.set_ylabel('Y')
                ax5.set_title('Target vs Best Result Curve Comparison')
                ax5.legend()
                ax5.grid(True, alpha=0.3)

            except Exception as e:
                ax5.text(0.5, 0.5, f'Error loading curves: {str(e)}',
                         ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Curve Comparison (Error)')

            plt.tight_layout()

            # Save figure
            save_path = self.results_dir / "optimization_progress.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization progress plot saved to: {save_path}")

            # 非阻塞显示图表
            plt.show(block=False)
            plt.pause(3)  # 显示3秒后继续
            plt.close()  # 关闭图表

        except Exception as e:
            print(f"Error in visualization: {e}")

    def run_optimization(self,
                         max_iterations: int = 10,
                         samples_per_iteration: int = 50,
                         sampling_strategy: str = "adaptive_grid",
                         convergence_tolerance: float = 1e-4,
                         pool_size: int = 44):
        """
        Run complete optimization workflow

        Args:
            max_iterations: Maximum number of iterations
            samples_per_iteration: Number of samples per iteration
            sampling_strategy: Sampling strategy
            convergence_tolerance: Convergence tolerance
            pool_size: Number of parallel processes
        """
        print("=" * 60)
        print("Starting iterative parameter optimization")
        print("=" * 60)
        print(f"Maximum iterations: {max_iterations}")
        print(f"Samples per iteration: {samples_per_iteration}")
        print(f"Sampling strategy: {sampling_strategy}")
        print(f"Convergence tolerance: {convergence_tolerance}")
        print(f"Parallel processes: {pool_size}")
        print("=" * 60)

        for iteration in range(1, max_iterations + 1):
            print(f"\n{'=' * 20} Iteration {iteration} {'=' * 20}")
            start_time = time.time()

            try:
                # 1. Generate parameter combinations
                print("Generating parameter combinations...")
                param_combinations = self.generate_parameter_combinations(
                    strategy=sampling_strategy,
                    n_samples=samples_per_iteration
                )
                print(f"Generated {len(param_combinations)} parameter combinations")

                # 2. Generate configuration files
                print("Generating configuration files...")
                generated_files, list_file = self.generate_config_files(
                    param_combinations, iteration
                )

                if not generated_files:
                    print("No configuration files generated, skipping iteration")
                    continue

                # 3. Run simulations
                print("Running simulations...")
                successful_sims = self.run_simulations(list_file, pool_size)

                if successful_sims == 0:
                    print("No successful simulations, skipping this iteration")
                    continue

                # 4. Analyze results
                print("Analyzing results...")
                best_params, best_distance, best_file = self.analyze_results(
                    generated_files, iteration
                )

                # 5. Record results
                computation_time = time.time() - start_time
                result = OptimizationResult(
                    iteration=iteration,
                    best_params=best_params,
                    best_distance=best_distance,
                    best_file=best_file,
                    convergence_history=[r.best_distance for r in self.optimization_history] + [best_distance],
                    param_history=[r.best_params for r in self.optimization_history] + [best_params],
                    total_evaluations=len(generated_files),
                    computation_time=computation_time
                )

                self.optimization_history.append(result)

                print(f"Iteration {iteration} completed:")
                print(f"  Best DTW distance: {best_distance:.6f}")
                print(f"  Computation time: {computation_time:.2f} seconds")
                print(f"  Number of evaluations: {len(generated_files)}")

                # 6. Update search ranges
                if iteration < max_iterations:
                    self.update_parameter_ranges(best_params)

                # 7. Check convergence
                if self.check_convergence(convergence_tolerance, min_iterations=5):
                    print(f"\nOptimization converged after iteration {iteration}!")
                    break

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                traceback.print_exc()  # 打印完整错误堆栈
                continue

        # Final results summary
        print("\n" + "=" * 60)
        print("Optimization completed!")
        print("=" * 60)

        if self.optimization_history:
            final_result = min(self.optimization_history, key=lambda x: x.best_distance)
            print(f"Global optimal result:")
            print(f"  Iteration: {final_result.iteration}")
            print(f"  Best DTW distance: {final_result.best_distance:.6f}")
            print(f"  Optimal parameters:")
            for param_name, value in final_result.best_params.items():
                if param_name in self.parameters:
                    desc = self.parameters[param_name].description
                    print(f"    {desc} ({param_name}): {value:.6f}")
            print(f"  Optimal configuration file: {final_result.best_file}")

            # Save final results
            try:
                final_results_file = self.results_dir / "final_optimization_results.json"
                with open(final_results_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'optimization_summary': {
                            'total_iterations': len(self.optimization_history),
                            'total_evaluations': sum(r.total_evaluations for r in self.optimization_history),
                            'total_time': sum(r.computation_time for r in self.optimization_history),
                            'convergence_achieved': self.check_convergence(convergence_tolerance)
                        },
                        'final_result': {
                            'iteration': final_result.iteration,
                            'best_distance': final_result.best_distance,
                            'best_params': final_result.best_params,
                            'best_file': final_result.best_file
                        },
                        'iteration_history': [asdict(r) for r in self.optimization_history]
                    }, f, indent=2)

                print(f"Detailed results saved to: {final_results_file}")
            except Exception as e:
                print(f"Warning: Failed to save final results: {e}")

            # Generate visualization
            self.visualize_optimization_progress()

        else:
            print("Optimization failed: no valid results")

        print("Program completed successfully!")


def main():
    """Main function example"""
    # Configuration paths (please modify according to actual situation)
    base_config_path = "/home/liu/桌面/amorsyn/data/carbon-fiber/00000.yaml"
    target_curve_file = "00000.txt"  # Target curve file name
    work_directory = "/home/liu/桌面/amorsyn/data/carbon-fiber"
    exec_file_path = "/home/liu/桌面/bullet-3d-printing/examples/3DPrinting/3DPrintingGui"

    # Create optimizer
    optimizer = IterativeParameterOptimizer(
        base_config_path=base_config_path,
        target_curve_file=target_curve_file,
        work_directory=work_directory,
        exec_file_path=exec_file_path
    )

    # Run optimization
    optimizer.run_optimization(
        max_iterations=8,
        samples_per_iteration=30,
        sampling_strategy="adaptive_grid",  # Options: "grid", "random", "adaptive_grid", "latin_hypercube"
        convergence_tolerance=1e-4,
        pool_size=44
    )

    print("All optimization tasks completed!")


if __name__ == "__main__":
    main()