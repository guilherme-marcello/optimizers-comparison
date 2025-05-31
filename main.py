import numpy as np
import pandas as pd
import time
import json
import os
import psutil # For memory usage
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars

# --- Configuration ---
N_FEATURES = 2048
N_TARGETS = 63
MAX_ITERATIONS = 2

# --- 1. Data Loader ---
class DataLoader:
    """
    Handles loading data from CSV files.
    Assumes CSV format: N_rows x (n_features + n_targets) columns.
    Each row: [feature_1, ..., feature_n, target_1, ..., target_m]
    """
    def __init__(self, n_features=N_FEATURES, n_targets=N_TARGETS):
        self.n_features = n_features
        self.n_targets = n_targets

    def load_data(self, csv_filepath):
        """
        Loads data from a CSV file.
        Returns:
            X (np.ndarray): Feature matrix (N_samples, n_features)
            Y (np.ndarray): Target matrix (N_samples, n_targets)
        """
        if not os.path.exists(csv_filepath):
            print(f"Warning: File {csv_filepath} not found. Returning None.")
            return None, None
        
        print(f"Loading data from {csv_filepath}...")
        try:
            df = pd.read_csv(csv_filepath, header=None)
            data = df.values.astype(np.float64)
            
            X = data[:, :self.n_features]
            Y = data[:, self.n_features:self.n_features + self.n_targets]
            
            print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")
            if X.shape[1] != self.n_features:
                raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
            if Y.shape[1] != self.n_targets:
                raise ValueError(f"Expected {self.n_targets} targets, got {Y.shape[1]}")
            
            return X, Y
        except Exception as e:
            print(f"Error loading data from {csv_filepath}: {e}")
            return None, None

# --- 2. Linear Regression Model ---
class LinearRegressionModel:
    """
    Defines the linear regression model and its mathematical properties.
    Objective: E(W) = ||Y - XW||_F^2
    """
    def __init__(self, n_features=N_FEATURES, n_targets=N_TARGETS):
        self.n_features = n_features
        self.n_targets = n_targets

    def initialize_weights(self, random_state=None):
        """Initializes weights W (n_features x n_targets) with small random values or zeros."""
        if random_state is not None:
            np.random.seed(random_state)
        # return np.random.randn(self.n_features, self.n_targets) * 0.01
        return np.zeros((self.n_features, self.n_targets), dtype=np.float64)


    def objective_function(self, X, Y, W):
        """Computes ||Y - XW||_F^2."""
        if X is None or Y is None or W is None: return np.inf
        residuals = Y - X @ W
        return np.sum(residuals**2) # Equivalent to np.linalg.norm(residuals, 'fro')**2

    def gradient(self, X, Y, W):
        """Computes gradient dE/dW = 2 * X^T @ (XW - Y)."""
        if X is None or Y is None or W is None: return np.zeros_like(W)
        return 2 * X.T @ (X @ W - Y)

    def hessian_term_xtx(self, X):
        """Computes X^T @ X, the core part of the Hessian for Newton's method."""
        if X is None: return None
        return X.T @ X

# --- 3. Optimizer Base Class ---
class Optimizer:
    """Base class for optimization algorithms."""
    def __init__(self, model, name="Optimizer", max_iter=MAX_ITERATIONS):
        self.model = model
        self.name = name
        self.max_iter = max_iter
        self.history = {
            "objective_values": [],
            "iteration_times": [],
            "memory_rss_bytes": [],
            "step_sizes": []
        }

    def _backtracking_line_search(self, X, Y, W, grad_W, pk, 
                                  c=0.1, beta=0.5, alpha_init=1.0,
                                  X_batch=None, Y_batch=None): # For SGD
        """
        Performs backtracking line search.
        W: current weights
        grad_W: gradient at W (full or mini-batch)
        pk: search direction
        X_batch, Y_batch: mini-batch data for SGD objective evaluation in line search
        """
        alpha = alpha_init
        
        # For SGD, evaluate objective and gradient term on the mini-batch
        eval_X, eval_Y = (X_batch, Y_batch) if X_batch is not None else (X, Y)

        current_objective = self.model.objective_function(eval_X, eval_Y, W)
        # The term grad_W.T @ pk for matrices is tr(grad_W.T @ pk)
        grad_pk_term = np.sum(grad_W * pk) # Element-wise product and sum

        for _ in range(30): # Max 30 steps for line search to prevent infinite loops
            W_new = W + alpha * pk
            new_objective = self.model.objective_function(eval_X, eval_Y, W_new)
            
            if new_objective <= current_objective + c * alpha * grad_pk_term:
                return alpha
            alpha *= beta
        return alpha # Return smallest alpha if condition not met (or 0 if too small)

    def optimize(self, X_train, Y_train, W_init):
        """Placeholder for the optimization logic."""
        raise NotImplementedError("Subclasses must implement the optimize method.")

    def get_history(self):
        return self.history

# --- 4. Specific Optimizer Implementations ---
class SteepestDescentOptimizer(Optimizer):
    def __init__(self, model, max_iter=MAX_ITERATIONS):
        super().__init__(model, name="SteepestDescent", max_iter=max_iter)

    def optimize(self, X_train, Y_train, W_init):
        W = W_init.copy()
        process = psutil.Process(os.getpid())
        self.history = {
            "objective_values": [], "iteration_times": [], 
            "memory_rss_bytes": [], "step_sizes": []
        }

        if X_train is None or Y_train is None:
            print(f"{self.name}: Training data missing, skipping optimization.")
            return W, self.history

        print(f"Starting {self.name} optimization...")
        for i in tqdm(range(self.max_iter), desc=self.name):
            iter_start_time = time.perf_counter()
            mem_start = process.memory_info().rss

            grad_W = self.model.gradient(X_train, Y_train, W)
            pk = -grad_W # Search direction

            # Stop if gradient is too small
            if np.linalg.norm(grad_W) < 1e-8:
                print(f"{self.name}: Gradient norm too small, stopping at iteration {i}.")
                break

            alpha = self._backtracking_line_search(X_train, Y_train, W, grad_W, pk, alpha_init=1.0)
            
            W += alpha * pk
            
            iter_end_time = time.perf_counter()
            mem_end = process.memory_info().rss

            self.history["objective_values"].append(self.model.objective_function(X_train, Y_train, W))
            self.history["iteration_times"].append(iter_end_time - iter_start_time)
            self.history["memory_rss_bytes"].append(mem_end) # Could also use mem_end - mem_start
            self.history["step_sizes"].append(alpha)
        
        print(f"{self.name} optimization finished.")
        return W, self.history

class StochasticGradientDescentOptimizer(Optimizer):
    def __init__(self, model, batch_size=128, max_iter=MAX_ITERATIONS):
        super().__init__(model, name="StochasticGradientDescent", max_iter=max_iter)
        self.batch_size = batch_size

    def optimize(self, X_train, Y_train, W_init):
        W = W_init.copy()
        process = psutil.Process(os.getpid())
        self.history = {
            "objective_values": [], "iteration_times": [],
            "memory_rss_bytes": [], "step_sizes": []
        }

        if X_train is None or Y_train is None:
            print(f"{self.name}: Training data missing, skipping optimization.")
            return W, self.history
            
        n_samples = X_train.shape[0]
        if n_samples == 0:
            print(f"{self.name}: No training samples, skipping optimization.")
            return W, self.history


        print(f"Starting {self.name} optimization...")
        for i in tqdm(range(self.max_iter), desc=self.name):
            iter_start_time = time.perf_counter()
            mem_start = process.memory_info().rss

            # Mini-batch sampling
            indices = np.random.permutation(n_samples)[:self.batch_size]
            X_batch, Y_batch = X_train[indices], Y_train[indices]

            grad_W_batch = self.model.gradient(X_batch, Y_batch, W)
            pk = -grad_W_batch

            if np.linalg.norm(grad_W_batch) < 1e-8: # Check batch gradient
                 # Evaluate full gradient to be sure
                full_grad_W = self.model.gradient(X_train, Y_train, W)
                if np.linalg.norm(full_grad_W) < 1e-7:
                    print(f"{self.name}: Full gradient norm too small, stopping at iteration {i}.")
                    break
                # else:
                #     print(f"{self.name}: Batch gradient small, but full gradient is {np.linalg.norm(full_grad_W)}. Continuing.")


            alpha = self._backtracking_line_search(X_train, Y_train, W, grad_W_batch, pk, 
                                                  alpha_init=0.1, # Smaller initial alpha for SGD
                                                  X_batch=X_batch, Y_batch=Y_batch) # Pass batch for line search objective
            
            W += alpha * pk
            
            iter_end_time = time.perf_counter()
            mem_end = process.memory_info().rss
            
            # Objective on full dataset for consistent tracking
            self.history["objective_values"].append(self.model.objective_function(X_train, Y_train, W))
            self.history["iteration_times"].append(iter_end_time - iter_start_time)
            self.history["memory_rss_bytes"].append(mem_end)
            self.history["step_sizes"].append(alpha)

        print(f"{self.name} optimization finished.")
        return W, self.history

class NewtonMethodOptimizer(Optimizer):
    def __init__(self, model, max_iter=MAX_ITERATIONS, regularization=1e-6):
        super().__init__(model, name="NewtonMethod", max_iter=max_iter)
        self.regularization = regularization # For XTX inversion
        self.XTX_inv = None # To cache (X^T X)^-1

    def optimize(self, X_train, Y_train, W_init):
        W = W_init.copy()
        process = psutil.Process(os.getpid())
        self.history = {
            "objective_values": [], "iteration_times": [],
            "memory_rss_bytes": [], "step_sizes": []
        }

        if X_train is None or Y_train is None:
            print(f"{self.name}: Training data missing, skipping optimization.")
            return W, self.history

        print(f"Starting {self.name} optimization...")
        
        # Precompute (X^T X + lambda I)^-1
        # This is the computationally intensive part for Newton's method setup
        xtx_setup_start = time.perf_counter()
        XTX = self.model.hessian_term_xtx(X_train)
        if XTX is None:
             print(f"{self.name}: XTX is None, cannot proceed.")
             return W, self.history
        
        identity = np.eye(XTX.shape[0]) * self.regularization
        try:
            # Using pseudo-inverse for stability if XTX is singular/ill-conditioned
            # self.XTX_inv = np.linalg.pinv(XTX + identity) 
            # Or, solve linear system for each column of search direction
            # P_k = W_sol - W_k, where W_sol = (XTX)^-1 @ XTY
            # XTY = X_train.T @ Y_train
            # W_sol_numerator = X_train.T @ Y_train
            # W_sol = np.linalg.solve(XTX + identity, W_sol_numerator)
            # This is for direct solution. For iterative Newton:
            # P_k such that (XTX)P_k = -grad_W
            # (XTX)P_k = X^T(Y - XW_k)
            # So P_k = (XTX)^-1 @ X^T @ (Y - XW_k)
            # This is W_sol - W_k
            # For stability, we compute LU decomposition of (XTX + identity)
            from scipy.linalg import lu_factor, lu_solve
            self.lu_XTX_plus_reg = lu_factor(XTX + identity)
            print(f"LU decomposition of (X^T X + reg*I) computed in {time.perf_counter() - xtx_setup_start:.2f}s")
        except np.linalg.LinAlgError as e:
            print(f"Error inverting/decomposing XTX for {self.name}: {e}. Optimizer may not work.")
            self.lu_XTX_plus_reg = None # Indicate failure
            # Fallback or stop
            return W, self.history


        for i in tqdm(range(self.max_iter), desc=self.name):
            iter_start_time = time.perf_counter()
            mem_start = process.memory_info().rss

            grad_W = self.model.gradient(X_train, Y_train, W)
            
            if np.linalg.norm(grad_W) < 1e-8:
                print(f"{self.name}: Gradient norm too small, stopping at iteration {i}.")
                break
            
            if self.lu_XTX_plus_reg is None:
                 print(f"{self.name}: XTX inverse/LU not available, cannot compute search direction. Stopping.")
                 break

            # Search direction P_k = (X^T X + lambda I)^-1 @ (X^T Y - X^T X W_k)
            # which is (X^T X + lambda I)^-1 @ (-grad_W / 2)  -- careful with factor of 2
            # grad_W = 2 * (X^T X W - X^T Y) = -2 * (X^T Y - X^T X W)
            # So, (X^T Y - X^T X W) = -grad_W / 2
            # P_k = solve (XTX + reg*I) P = -grad_W / 2
            
            # pk_numerator = X_train.T @ (Y_train - X_train @ W) # This is -grad_W / 2
            pk_numerator = -grad_W / 2.0

            pk_cols = []
            for col_idx in range(pk_numerator.shape[1]):
                pk_col = lu_solve(self.lu_XTX_plus_reg, pk_numerator[:, col_idx], trans=0)
                pk_cols.append(pk_col)
            pk = np.stack(pk_cols, axis=1)

            alpha = self._backtracking_line_search(X_train, Y_train, W, grad_W, pk, alpha_init=1.0)
            
            W += alpha * pk
            
            iter_end_time = time.perf_counter()
            mem_end = process.memory_info().rss

            self.history["objective_values"].append(self.model.objective_function(X_train, Y_train, W))
            self.history["iteration_times"].append(iter_end_time - iter_start_time)
            self.history["memory_rss_bytes"].append(mem_end)
            self.history["step_sizes"].append(alpha)
        
        print(f"{self.name} optimization finished.")
        return W, self.history

class GaussNewtonMethodOptimizer(NewtonMethodOptimizer): # Inherits from Newton
    def __init__(self, model, max_iter=MAX_ITERATIONS, regularization=1e-6):
        # For linear least squares, Gauss-Newton is identical to Newton's method.
        # The Hessian approximation J^T J is exactly X^T X (ignoring factor of 2 which cancels).
        super().__init__(model, max_iter=max_iter, regularization=regularization)
        self.name = "GaussNewtonMethod"


# --- 5. Benchmark Runner ---
class BenchmarkRunner:
    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model
        self.benchmark_results = {"dataset_info": {}, "optimizer_results": []}

    def _calculate_mse(self, X, Y, W):
        if X is None or Y is None or W is None: return np.inf
        predictions = X @ W
        errors = Y - predictions
        mse_per_target = np.mean(errors**2, axis=0) # MSE for each of m targets
        return np.mean(mse_per_target) # Average MSE across all targets

    def run(self, optimizers, train_csv_path, test_csv_path, W_init_seed=42):
        X_train, Y_train = self.data_loader.load_data(train_csv_path)
        X_test, Y_test = self.data_loader.load_data(test_csv_path)

        self.benchmark_results["dataset_info"] = {
            "n_features": self.model.n_features,
            "n_targets": self.model.n_targets,
            "n_train_samples": X_train.shape[0] if X_train is not None else 0,
            "n_test_samples": X_test.shape[0] if X_test is not None else 0,
            "train_csv": train_csv_path,
            "test_csv": test_csv_path
        }

        if X_train is None or Y_train is None:
            print("Cannot run benchmark due to missing training data.")
            return

        W_initial = self.model.initialize_weights(random_state=W_init_seed)

        for optimizer in optimizers:
            print(f"\n--- Running Optimizer: {optimizer.name} ---")
            W_final, history = optimizer.optimize(X_train, Y_train, W_initial.copy())
            
            test_mse = self._calculate_mse(X_test, Y_test, W_final)
            total_time = np.sum(history.get("iteration_times", [0]))

            opt_result = {
                "optimizer_name": optimizer.name,
                "W_final_norm": float(np.linalg.norm(W_final)) if W_final is not None else None, # Avoid saving large W
                "test_mse": float(test_mse) if test_mse is not None else None,
                "total_optimization_time": float(total_time),
                "iterations_data": [
                    {
                        "iter": i + 1,
                        "objective": float(history["objective_values"][i]) if i < len(history["objective_values"]) else None,
                        "time": float(history["iteration_times"][i]) if i < len(history["iteration_times"]) else None,
                        "memory_rss_bytes": int(history["memory_rss_bytes"][i]) if i < len(history["memory_rss_bytes"]) else None,
                        "step_size": float(history["step_sizes"][i]) if i < len(history["step_sizes"]) else None
                    } for i in range(len(history.get("objective_values", []))) # Use actual number of iterations run
                ]
            }
            if hasattr(optimizer, 'batch_size'): # For SGD
                opt_result['batch_size'] = optimizer.batch_size
            if hasattr(optimizer, 'regularization'): # For Newton/GN
                opt_result['regularization'] = optimizer.regularization

            self.benchmark_results["optimizer_results"].append(opt_result)
        
        return self.benchmark_results

    def plot_results(self):
        if not self.benchmark_results["optimizer_results"]:
            print("No optimizer results to plot.")
            return

        plt.figure(figsize=(12, 8))
        for result in self.benchmark_results["optimizer_results"]:
            obj_values = [item['objective'] for item in result['iterations_data'] if item['objective'] is not None]
            if obj_values: # Only plot if there are objective values
                 # Cap iterations at MAX_ITERATIONS for plotting if optimizer stopped early
                iters_to_plot = min(len(obj_values), MAX_ITERATIONS)
                plt.plot(range(1, iters_to_plot + 1), obj_values[:iters_to_plot], label=result["optimizer_name"])
        
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value (log scale)")
        plt.title("Objective Function Progression")
        plt.yscale('log') # Objective can decrease by orders of magnitude
        plt.legend()
        plt.grid(True, which="both", ls="--")
        
        plot_filename = "objective_progression.png"
        plt.savefig(plot_filename)
        print(f"Saved objective progression plot to {plot_filename}")
        plt.show()


    def export_to_json(self, filepath="benchmark_results.json"):
        print(f"Exporting benchmark results to {filepath}...")
        try:
            with open(filepath, 'w') as f:
                json.dump(self.benchmark_results, f, indent=4)
            print("Export successful.")
        except Exception as e:
            print(f"Error exporting to JSON: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure these files are in the same directory as the script or provide full paths.
    train_file = "Trn.csv"
    test_file = "Tst.csv"
    current_n_features = N_FEATURES
    current_n_targets = N_TARGETS
    # --- End Configuration ---

    data_loader = DataLoader(n_features=current_n_features, n_targets=current_n_targets)
    model = LinearRegressionModel(n_features=current_n_features, n_targets=current_n_targets)

    # Initialize optimizers with the model
    optimizers_to_run = [
        SteepestDescentOptimizer(model, max_iter=MAX_ITERATIONS),
        StochasticGradientDescentOptimizer(model, batch_size=128, max_iter=MAX_ITERATIONS), # TODO: batch size can be tuned
        NewtonMethodOptimizer(model, max_iter=MAX_ITERATIONS, regularization=1e-5),
        GaussNewtonMethodOptimizer(model, max_iter=MAX_ITERATIONS, regularization=1e-5)
    ]

    # Run benchmark
    runner = BenchmarkRunner(data_loader, model)
    benchmark_data = runner.run(optimizers_to_run, train_file, test_file, W_init_seed=42)

    # Plot and export results
    if benchmark_data and benchmark_data["optimizer_results"]:
        runner.plot_results()
        runner.export_to_json("benchmark_results.json")
    else:
        print("Benchmarking did not produce results. Skipping plot and export.")

    print("\nBenchmark finished. Check 'benchmark_results.json' and 'objective_progression.png'.")