import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import json
import os
import psutil # For memory usage
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars
from scipy.linalg import lu_factor, lu_solve # For Newton's method


MAX_ITERATIONS = 500


# -- Step to load student_habits_performance.csv and generate its Trn.csv and Tst.csv --
def split_student_habits_data(input_csv, train_csv='Trn_student.csv', test_csv='Tst_student.csv', test_size=0.2, random_state=42):
    """
    Splits the student habits dataset into training and testing sets.
    Saves the training set to 'Trn_student.csv' and the testing set to 'Tst_student.csv'.
    """
    df = pd.read_csv(input_csv)
    df.info()
    numeric_df = df.select_dtypes(include=['number'])
    print("\nSelected Numeric Features:")
    print(list(numeric_df.columns))

    print(f"\nSplitting data into training and testing sets ({int((1-test_size)*100)}%/{int(test_size*100)}%)...")
    train_df, test_df = train_test_split(
        numeric_df,
        test_size=test_size,
        random_state=random_state
    )
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")

    train_df.to_csv(train_csv, index=False, header=False)
    test_df.to_csv(test_csv, index=False, header=False)

split_student_habits_data('student_habits_performance.csv')



class Dataset:
    """
    Represents a dataset with features and targets.
    Assumes CSV format: N_rows x (n_features + n_targets) columns.
    Each row: [feature_1, ..., feature_n, target_1, ..., target_m]
    """
    def __init__(self, name, train_csv, test_csv, n_features, n_targets):
        self.name = name
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.n_features = n_features
        self.n_targets = n_targets

# --- 1. Data Loader ---
class DataLoader:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def load_train_data(self):
        return self.load_data(self.dataset.train_csv)
    
    def load_test_data(self):
        return self.load_data(self.dataset.test_csv)

    def load_data(self, csv_filepath):
        """
        Loads data from a CSV file.
        Returns:
            X (np.ndarray): Feature matrix (n_features, N_samples)
            Y (np.ndarray): Target matrix (n_targets, N_samples)
        """
        if not os.path.exists(csv_filepath):
            print(f"Warning: File {csv_filepath} not found. Returning None.")
            return None, None
        
        print(f"Loading data from {csv_filepath}...")
        try:
            df = pd.read_csv(csv_filepath, header=None)
            data = df.values.astype(np.float64)
            
            # Transpose to get (n_features, N_samples) and (n_targets, N_samples)
            X = data[:, :self.dataset.n_features].T 
            Y = data[:, self.dataset.n_features:self.dataset.n_features + self.dataset.n_targets].T
            
            print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")
            if X.shape[0] != self.dataset.n_features:
                raise ValueError(f"Expected {self.dataset.n_features} features (X.shape[0]), got {X.shape[0]}")
            if Y.shape[0] != self.dataset.n_targets:
                raise ValueError(f"Expected {self.dataset.n_targets} targets (Y.shape[0]), got {Y.shape[0]}")
            if X.shape[1] != Y.shape[1] and X.shape[1] != 0: # Number of samples must match
                 raise ValueError(f"Number of samples in X ({X.shape[1]}) and Y ({Y.shape[1]}) must match.")

            return X, Y
        except Exception as e:
            print(f"Error loading data from {csv_filepath}: {e}")
            return None, None

# --- 2. Linear Regression Model ---
class LinearRegressionModel:
    """
    Defines the linear regression model and its mathematical properties.
    W is (n_features, n_targets)
    X is (n_features, N_samples)
    Y is (n_targets, N_samples)
    Objective: E(W) = ||W^T X - Y||_F^2
    """
    def __init__(self, n_features, n_targets):
        self.n_features = n_features
        self.n_targets = n_targets

    def initialize_weights(self, method='random', random_state=42, scale=0.01):
        """
        Initializes weights W (n_features x n_targets).
        Supported methods:
            - 'random': Random normal initialization scaled by `scale`.
            - 'zeros': All weights set to zero.
        """
        if method == 'random':
            np.random.seed(random_state)
            return np.random.randn(self.n_features, self.n_targets) * scale
        elif method == 'zeros':
            return np.zeros((self.n_features, self.n_targets), dtype=np.float64)
        else:
            raise ValueError(f"Unknown weight initialization method: '{method}'")

    def objective_function(self, X, Y, W):
        """Computes E(W) = ||W^T X - Y||_F^2."""
        if X is None or Y is None or W is None: return np.inf
        # W.T (m, n) @ X (n, N) -> (m, N)
        # Y (m, N)
        residuals = W.T @ X - Y 
        return np.sum(residuals**2)

    def gradient(self, X, Y, W):
        """Computes gradient dE/dW = 2 * X @ (W^T @ X - Y)^T."""
        if X is None or Y is None or W is None: 
            return np.zeros((self.n_features, self.n_targets), dtype=np.float64)
        # W.T (m, n) @ X (n, N) -> (m, N)
        # Y (m, N)
        # residuals_term = W.T @ X - Y (m, N)
        # residuals_term.T (N, m)
        # X (n, N) @ residuals_term.T (N, m) -> (n, m)
        residuals_term = W.T @ X - Y
        return 2 * X @ residuals_term.T

    def hessian_term_xx_t(self, X):
        """Computes X @ X^T, the core part of the Hessian for Newton's method. (n x n)"""
        if X is None: return None
        # X (n, N) @ X.T (N, n) -> (n, n)
        return X @ X.T

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
                                  c=0.1, tau=0.5, alpha_init=1.0e-2,
                                  X_batch=None, Y_batch=None): # For SGD
        """
        Performs backtracking line search.
        W: current weights (n, m)
        grad_W: gradient at W (n, m) (full or mini-batch)
        pk: search direction (n, m)
        X_batch, Y_batch: mini-batch data for SGD objective evaluation in line search
                          X_batch (n, batch_size), Y_batch (m, batch_size)
        """
        alpha = alpha_init
        
        eval_X, eval_Y = (X_batch, Y_batch) if X_batch is not None else (X, Y)

        current_objective = self.model.objective_function(eval_X, eval_Y, W)
        # The term grad_W.T @ pk for matrices is tr(grad_W.T @ pk)
        grad_pk_term = np.sum(grad_W * pk) # Element-wise product and sum

        for _ in range(30): # Max 30 steps for line search
            W_new = W + alpha * pk
            new_objective = self.model.objective_function(eval_X, eval_Y, W_new)
            
            # Wolfe condition
            if new_objective <= current_objective + c * alpha * grad_pk_term:
                return alpha
            alpha *= tau
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
            mem_start = process.memory_info().rss # Memory per iter can be noisy

            grad_W = self.model.gradient(X_train, Y_train, W)
            pk = -grad_W # Search direction

            # Stop if gradient is too small
            if np.linalg.norm(grad_W) < 1e-8:
                print(f"{self.name}: Gradient norm too small, stopping at iteration {i}.")
                break

            alpha = self._backtracking_line_search(X_train, Y_train, W, grad_W, pk, alpha_init=1e-2)
            W += alpha * pk
            
            iter_end_time = time.perf_counter()
            mem_end = process.memory_info().rss

            self.history["objective_values"].append(self.model.objective_function(X_train, Y_train, W))
            self.history["iteration_times"].append(iter_end_time - iter_start_time)
            self.history["memory_rss_bytes"].append(mem_end) # TODO: use mem_end - mem_start ?
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
            
        n_total_samples = X_train.shape[1] # N_samples is the second dimension
        if n_total_samples == 0:
            print(f"{self.name}: No training samples, skipping optimization.")
            return W, self.history

        print(f"Starting {self.name} optimization...")
        for i in tqdm(range(self.max_iter), desc=self.name):
            iter_start_time = time.perf_counter()
            mem_start = process.memory_info().rss

            # Mini-batch sampling: select columns (samples)
            indices = np.random.permutation(n_total_samples)[:self.batch_size]
            X_batch, Y_batch = X_train[:, indices], Y_train[:, indices]

            grad_W_batch = self.model.gradient(X_batch, Y_batch, W)
            pk = -grad_W_batch

            # Stopping condition based on batch gradient can be tricky.
            # Optional: Check full gradient norm periodically if batch gradient is small.
            if np.linalg.norm(grad_W_batch) < 1e-7: # Looser tolerance for batch grad
                full_grad_W = self.model.gradient(X_train, Y_train, W)
                if np.linalg.norm(full_grad_W) < 1e-7:
                    print(f"{self.name}: Full gradient norm too small, stopping at iteration {i}.")
                    break

            alpha = self._backtracking_line_search(X_train, Y_train, W, grad_W_batch, pk, 
                                                  alpha_init=0.1, # Smaller initial alpha for SGD
                                                  X_batch=X_batch, Y_batch=Y_batch) 
            
            W += alpha * pk
            
            iter_end_time = time.perf_counter()
            mem_end = process.memory_info().rss
            
            self.history["objective_values"].append(self.model.objective_function(X_train, Y_train, W))
            self.history["iteration_times"].append(iter_end_time - iter_start_time)
            self.history["memory_rss_bytes"].append(mem_end)
            self.history["step_sizes"].append(alpha)

        print(f"{self.name} optimization finished.")
        return W, self.history

class NewtonMethodOptimizer(Optimizer):
    def __init__(self, model, max_iter=MAX_ITERATIONS, regularization=1e-6):
        super().__init__(model, name="NewtonMethod", max_iter=max_iter)
        self.regularization = regularization 
        self.lu_XXT_plus_reg = None # To cache LU decomposition of (X X^T + lambda I)

    def optimize(self, X_train, Y_train, W_init):
        W = W_init.copy()
        process = psutil.Process(os.getpid())
        self.history = {
            "objective_values": [], "iteration_times": [],
            "memory_rss_bytes": [], "step_sizes": []
        }

        print(f"Starting {self.name} optimization...")
        
        # Precompute LU decomposition of (X X^T + lambda I)
        # X X^T is (n_features x n_features)
        xxt_setup_start = time.perf_counter()
        XXT = self.model.hessian_term_xx_t(X_train) # This is X @ X.T
        if XXT is None:
             print(f"{self.name}: XXT is None, cannot proceed.")
             return W, self.history
        
        identity = np.eye(XXT.shape[0]) * self.regularization
        try:
            self.lu_XXT_plus_reg = lu_factor(XXT + identity)
            print(f"LU decomposition of (X X^T + reg*I) computed in {time.perf_counter() - xxt_setup_start:.2f}s")
        except np.linalg.LinAlgError as e:
            print(f"Error decomposing (X X^T + reg*I) for {self.name}: {e}. Optimizer may not work.")
            self.lu_XXT_plus_reg = None
            return W, self.history

        for i in tqdm(range(self.max_iter), desc=self.name):
            iter_start_time = time.perf_counter()
            mem_start = process.memory_info().rss

            grad_W = self.model.gradient(X_train, Y_train, W)
            
            if np.linalg.norm(grad_W) < 1e-8:
                print(f"{self.name}: Gradient norm too small, stopping at iteration {i}.")
                break
            
            if self.lu_XXT_plus_reg is None:
                 print(f"{self.name}: LU decomposition not available. Stopping.")
                 break

            # Search direction P_k solves (X X^T + lambda I) P_k = - (Gradient_W / 2)
            # Gradient_W = 2 * X @ (W^T @ X - Y)^T
            # So, - (Gradient_W / 2) = - X @ (W^T @ X - Y)^T
            pk_numerator = -X_train @ (W.T @ X_train - Y_train).T

            pk_cols = []
            for col_idx in range(pk_numerator.shape[1]): # Iterate over m target dimensions
                pk_col = lu_solve(self.lu_XXT_plus_reg, pk_numerator[:, col_idx], trans=0)
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


class LBFGSOptimizer(Optimizer):
    def __init__(self, model, max_iter=MAX_ITERATIONS, memory=10):
        super().__init__(model, name="L-BFGS", max_iter=max_iter)
        self.memory = memory  # History size (m in L-BFGS)

    def optimize(self, X_train, Y_train, W_init):
        W = W_init.copy()
        s_list = []  # List of s = W_{k+1} - W_k
        y_list = []  # List of y = grad_{k+1} - grad_k
        rho_list = []

        process = psutil.Process(os.getpid())
        self.history = {
            "objective_values": [], "iteration_times": [],
            "memory_rss_bytes": [], "step_sizes": []
        }

        print(f"Starting {self.name} optimization...")

        for i in tqdm(range(self.max_iter), desc=self.name):
            iter_start_time = time.perf_counter()
            mem_start = process.memory_info().rss

            grad = self.model.gradient(X_train, Y_train, W)
            q = grad.flatten()

            # === Two-loop recursion ===
            alpha_list = []
            for s, y, rho in reversed(list(zip(s_list, y_list, rho_list))):
                alpha = rho * np.dot(s, q)
                alpha_list.append(alpha)
                q = q - alpha * y

            if y_list:
                last_y = y_list[-1]
                last_s = s_list[-1]
                gamma = np.dot(last_s, last_y) / np.dot(last_y, last_y)
            else:
                gamma = 1.0
            r = gamma * q

            for s, y, rho, alpha in zip(s_list, y_list, rho_list, reversed(alpha_list)):
                beta = rho * np.dot(y, r)
                r = r + s * (alpha - beta)

            direction = -r.reshape(W.shape)
            alpha_step = self._backtracking_line_search(X_train, Y_train, W, grad, direction, alpha_init=1.0)
            W_new = W + alpha_step * direction
            grad_new = self.model.gradient(X_train, Y_train, W_new)

            s = (W_new - W).flatten()
            y = (grad_new - grad).flatten()

            ys = np.dot(y, s)
            if ys > 1e-10:  # Maintain curvature condition
                rho = 1.0 / ys
                s_list.append(s)
                y_list.append(y)
                rho_list.append(rho)
                if len(s_list) > self.memory:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)

            W = W_new

            iter_end_time = time.perf_counter()
            mem_end = process.memory_info().rss

            self.history["objective_values"].append(self.model.objective_function(X_train, Y_train, W))
            self.history["iteration_times"].append(iter_end_time - iter_start_time)
            self.history["memory_rss_bytes"].append(mem_end)
            self.history["step_sizes"].append(alpha_step)

        print(f"{self.name} optimization finished.")
        return W, self.history


# --- 5. Benchmark Runner ---
class BenchmarkRunner:
    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model
        self.benchmark_results = {"dataset_info": {}, "optimizer_results": []}

    def _calculate_mse(self, X, Y, W):
        """
        Calculates Mean Squared Error.
        X (n, N_test), Y (m, N_test), W (n, m)
        Predictions: W.T @ X  (m, N_test)
        """
        if X is None or Y is None or W is None: return np.inf
        if X.shape[1] == 0: return np.inf # No test samples

        predictions = W.T @ X # (m, n) @ (n, N_test) -> (m, N_test)
        errors = Y - predictions # (m, N_test)
        
        # MSE per target (mean across samples for each target dimension)
        mse_per_target = np.mean(errors**2, axis=1) # (m,)
        return np.mean(mse_per_target) # Average MSE across all targets

    def run(self, optimizers, W_init_seed=42):
        X_train, Y_train = self.data_loader.load_train_data()
        X_test, Y_test = self.data_loader.load_test_data()

        self.benchmark_results["dataset_info"] = {
            "dataset_name": dataset.name,
            "n_features": self.model.n_features,
            "n_targets": self.model.n_targets,
            "n_train_samples": X_train.shape[1] if X_train is not None else 0,
            "n_test_samples": X_test.shape[1] if X_test is not None else 0,
            "train_csv": dataset.train_csv,
            "test_csv": dataset.test_csv
        }

        # for each weight initialization method, run the optimizers
        for weight_init_method in ['random', 'zeros']:
            print(f"Initializing weights using method: {weight_init_method}")
            W_initial = self.model.initialize_weights(method=weight_init_method, random_state=W_init_seed)
            # for each optimizer
            for optimizer in optimizers:
                print(f"\n--- Running Optimizer: {optimizer.name}, W_0 as {weight_init_method} ---")
                W_final, history = optimizer.optimize(X_train, Y_train, W_initial.copy())
                
                test_mse = self._calculate_mse(X_test, Y_test, W_final)
                total_time = np.sum(history.get("iteration_times", [0]))

                opt_result = {
                    "dataset_name": self.benchmark_results["dataset_info"]["dataset_name"],
                    "optimizer_name": f"{optimizer.name}+{weight_init_method}",
                    "W_final_norm": float(np.linalg.norm(W_final)) if W_final is not None else None,
                    "test_mse": float(test_mse) if np.isfinite(test_mse) else None,
                    "total_optimization_time_s": float(total_time),
                    "num_iterations_run": len(history.get("objective_values", [])),
                    "iterations_data": [
                        {
                            "iter": i + 1,
                            "objective": float(history["objective_values"][i]) if i < len(history["objective_values"]) else None,
                            "time_s": float(history["iteration_times"][i]) if i < len(history["iteration_times"]) else None,
                            "memory_rss_bytes": int(history["memory_rss_bytes"][i]) if i < len(history["memory_rss_bytes"]) else None,
                            "step_size": float(history["step_sizes"][i]) if i < len(history["step_sizes"]) else None
                        } for i in range(len(history.get("objective_values", [])))
                    ]
                }
                if hasattr(optimizer, 'batch_size'): # For SGD
                    opt_result['batch_size'] = optimizer.batch_size
                if hasattr(optimizer, 'regularization'): # For Newton/GN
                    opt_result['regularization'] = optimizer.regularization

                self.benchmark_results["optimizer_results"].append(opt_result)
        
        return self.benchmark_results

    def plot_results(self):
        plt.figure(figsize=(12, 8))
        sorted_results = sorted(self.benchmark_results["optimizer_results"], key=lambda r: r['optimizer_name'])
        for result in sorted_results:
            obj_values = [item['objective'] for item in result.get('iterations_data', []) if item.get('objective') is not None]
            if obj_values:
                iters_to_plot = min(len(obj_values), result.get("num_iterations_run", MAX_ITERATIONS))
                plt.plot(range(1, iters_to_plot + 1), obj_values[:iters_to_plot], label=result["optimizer_name"], marker='.', linestyle='-')
        
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value (log scale)")
        plt.title(f"Objective Function Progression for {sorted_results[0]['dataset_name']}")
        plt.yscale('log') 
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

def run_benchmark(dataset: Dataset):
    data_loader = DataLoader(dataset)
    model = LinearRegressionModel(n_features=dataset.n_features, n_targets=dataset.n_targets)

    # Initialize optimizers with the model
    optimizers_to_run = [
        SteepestDescentOptimizer(model, max_iter=MAX_ITERATIONS),
        StochasticGradientDescentOptimizer(model, batch_size=128, max_iter=MAX_ITERATIONS), # TODO: batch size can be tuned
        NewtonMethodOptimizer(model, max_iter=MAX_ITERATIONS, regularization=1e-5),
        LBFGSOptimizer(model, max_iter=MAX_ITERATIONS)
    ]

    # Run benchmark
    runner = BenchmarkRunner(data_loader, model)
    benchmark_data = runner.run(optimizers_to_run, W_init_seed=42)

    # Plot and export results
    if benchmark_data and benchmark_data["optimizer_results"]:
        runner.plot_results()
        runner.export_to_json("benchmark_results.json")
    else:
        print("Benchmarking did not produce results. Skipping plot and export.")

    print("\nBenchmark finished. Check 'benchmark_results.json' and 'objective_progression.png'.")

# --- Main Execution ---
if __name__ == "__main__":
    # Instanciate the dataset with configuration parameters. 

    hand_tracking_dataset = Dataset(
        name="ICVL hand tracking dataset",
        train_csv="Trn.csv",
        test_csv="Tst.csv",
        n_features=2048,
        n_targets=63
    )

    student_performance_dataset = Dataset(
        name="Student Habits and Performance",
        train_csv="Trn_student.csv",
        test_csv="Tst_student.csv",
        n_features=7, # age,study_hours_per_day,social_media_hours,netflix_hours,attendance_percentage,sleep_hours,exercise_frequency
        n_targets=2 # mental_health_rating,exam_score
    )
    # --- End Configuration ---

    for dataset in [student_performance_dataset]:
        print(f"\nRunning benchmark for dataset: {dataset.name}")
        run_benchmark(dataset)


