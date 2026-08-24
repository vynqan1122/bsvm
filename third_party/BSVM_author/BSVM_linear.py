# -*- coding: utf-8 -*-
"""
Verbose-traceable BSVM with Linear Kernel - Manuscript Figure Generator
Now includes:
 • Fixes visualization logic to show the correct boundary for each attempt.
 • Saves all plots as sequentially numbered, high-quality PNG files.
 • Removes titles from plots for clean inclusion in documents.
 • Adjusts figure size and font for better manuscript presentation.
 • Retains guaranteed convergence and detailed linear kernel logging.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# --- Global Settings for Manuscript-Ready Plots ---
plt.rcParams.update({'font.size': 12}) # Consistent font size
FIG_COUNT = 0 # Global counter for unique figure filenames

# ============================================================
# Helper Functions
# ============================================================

def compute_margin_width(model):
    """Return total distance between SVM margin lines (2 / ||w||)."""
    if hasattr(model, 'coef_'):
        w = model.coef_[0]
        return 2 / np.linalg.norm(w)
    return np.nan

def print_decision_boundary_info(model, X, label=""):
    """Prints weight vector, bias, and w^T x + b for samples."""
    if not hasattr(model, 'coef_'):
        print(f"\n📈 {label} Decision Boundary Info (Non-linear kernel)")
        return
    
    w = model.coef_[0]
    b = model.intercept_[0]
    decision_values = np.dot(X, w) + b
    print(f"\n📈 {label} Decision Boundary Info")
    print(f"  w = [{w[0]:.4f}, {w[1]:.4f}]")
    print(f"  b = {b:.4f}")
    print(f"  Sample decision values (first 10): {decision_values[:10]}")
    print(f"  Mean: {np.mean(decision_values):.4f}, Std: {np.std(decision_values):.4f}")
    return decision_values

def extend_samples(Xnew, ynew, X, y, h, yh, pipeline, weights=None):
    print("\n🔹 Entering extend_samples()")
    print(f"  Current training set size: {len(Xnew)}")
    print(f"  Remaining candidate pool size: {len(h)}")

    s = len(h)

    pipeline.set_params(class_weight={-1: 1.0, 1.0: 1.0})
    pipeline.fit(Xnew, ynew)
    
    margin_width = compute_margin_width(pipeline)
    print(f"  ✅ Model fitted on current Xnew/ynew | 📏 Margin width: {margin_width:.4f}")
    print_decision_boundary_info(pipeline, Xnew, label="Start of Iteration")
    
    y_pred = pipeline.predict(h)
    print(f"  Predictions on candidate pool (first 10): {y_pred[:10]}...")

    a, b = [], {}
    print("\n🧮 Computing priorities for candidate samples:")
    for i in range(s):
        decision_value = pipeline.decision_function([h[i]])[0]
        distance = abs(decision_value)
        if y_pred[i] == yh[i] and yh[i] * decision_value >= 1.0:
            a.append(i)
        else:
            priority = 1.0 / (distance + 1e-6)
            b[i] = priority
            coord = h[i]
            print(f"    🧭 Candidate {i:02d} | Coord=({coord[0]:.3f}, {coord[1]:.3f}) "
                  f"| Distance={distance:.4f} | Priority={priority:.4f}")

    print(f"\n  Candidate indices strongly classified (a): {a}")
    print(f"  Weak/misclassified candidates (b keys): {list(b.keys())}")

    while len(h) > 0:
        sorted_b = sorted(b, key=b.get, reverse=True)
        for idx in sorted_b:
            if idx not in b or idx >= len(h):
                continue

            candidate_coord, candidate_label = h[idx], yh[idx][0]
            print(f"\n  ➕ Attempting to add sample index {idx} "
                  f"({candidate_coord[0]:.3f}, {candidate_coord[1]:.3f}) label={int(candidate_label)}")

            plot_decision_boundary(pipeline, Xnew, ynew,
                         filename_base="attempt",
                         attempt_coord=candidate_coord, attempt_label=candidate_label)

            Xnew_temp = np.vstack((Xnew, candidate_coord))
            ynew_temp = np.hstack((ynew, candidate_label))

            pipeline.set_params(class_weight={-1: 1.0, 1.0: 1.0})
            pipeline.fit(Xnew_temp, ynew_temp)

            margin_width = compute_margin_width(pipeline)
            print(f"  ✅ Model re-fitted with {len(Xnew_temp)} samples | 📏 Margin width: {margin_width:.4f}")
            print_decision_boundary_info(pipeline, Xnew_temp, label="During Iteration")
            
            y_pred_Xnew = pipeline.predict(Xnew_temp)
            mis_idx = np.where(y_pred_Xnew != ynew_temp)[0]
            misclassified_flag = len(mis_idx) > 0
            print(f"  Misclassified? {misclassified_flag}")

            if misclassified_flag:
                print(f"  ❌ Adding sample ({candidate_coord[0]:.3f}, {candidate_coord[1]:.3f}) caused misclassification — rejecting.")
                plot_decision_boundary(pipeline, Xnew_temp, ynew_temp,
                                       filename_base="misclassification",
                                       mis_idx=mis_idx)
                
                # --- THE FIX: Revert the model to its state before this failed attempt ---
                pipeline.fit(Xnew, ynew)
                
                del b[idx]
                continue

            print(f"  ✅ Sample accepted ({candidate_coord[0]:.3f}, {candidate_coord[1]:.3f}) | New Margin: {margin_width:.4f}")
            
            Xnew = Xnew_temp
            ynew = ynew_temp
            h = np.delete(h, idx, axis=0)
            yh = np.delete(yh, idx, axis=0)
            
            print_decision_boundary_info(pipeline, Xnew, label="End of Iteration")
            plot_decision_boundary(pipeline, Xnew, ynew, filename_base="stage_accepted")
            
            progress_made = True
            return Xnew, ynew, h, yh, progress_made
        break

    print("  No more candidates could be added without causing misclassification.")
    progress_made = False
    return Xnew, ynew, h, yh, progress_made


def masterproblem(Xnew, ynew, X, y, h, yh, pipeline, weights):
    print("\n🚀 Starting masterproblem()")
    iteration = 0
    while len(h) > 0:
        iteration += 1
        print(f"\n==============================")
        print(f"🧩 Iteration {iteration}")
        print(f"Current training set size: {len(Xnew)} | Remaining: {len(h)}")

        pipeline.set_params(class_weight={-1: 1.0, 1.0: 1.0})
        Xnew, ynew, h, yh, progress_made = extend_samples(Xnew, ynew, X, y, h, yh, pipeline, weights)

        print(f"🔸 End of iteration {iteration}")
        print(f"    Xnew shape: {Xnew.shape} | ynew shape: {ynew.shape} | Remaining pool: {len(h)}")

        if not progress_made:
            print("🛑 No further samples could be added. Terminating.")
            break
            
    print("✅ masterproblem() completed.")


def Initialsolution(X, y, pipeline):
    print("\n🔹 Computing Initial Solution")
    X, y = X.copy(), y.copy()

    pipeline.set_params(class_weight={-1: 1.0, 1.0: 1.0})
    pipeline.fit(X, y)

    y_pred = pipeline.predict(X)
    i2 = (y_pred != y)
    d = y * pipeline.decision_function(X)
    i1 = d < 1
    h = np.logical_or(i1, i2)
    yh = y[h].reshape(-1, 1)
    print(f"  Initial support set: {np.sum(~h)}, Candidate pool: {np.sum(h)}")
    return X[~h], y[~h], X[h], yh


# ============================================================
# Visualization Helpers
# ============================================================

def plot_decision_boundary(model, X, y, filename_base, h=None, yh=None, attempt_coord=None, attempt_label=None, mis_idx=None):
    global FIG_COUNT
    FIG_COUNT += 1
    
    plt.figure(figsize=(4.5, 4))
    
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color="blue", alpha=0.6, label="Class -1")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", alpha=0.6, label="Class +1")

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    if hasattr(model, "decision_function"):
        Z = model.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
                   linestyles=['--', '-', '--'])
    
    if hasattr(model, "support_vectors_"):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=120,
                   facecolors='none', edgecolors='k', linewidths=2, label='Support Vectors')

    if h is not None and yh is not None:
        plt.scatter(h[yh.flatten()==-1,0], h[yh.flatten()==-1,1], s=200, facecolors='none', edgecolors='blue', marker='s', linewidths=2)
        plt.scatter(h[yh.flatten()==1,0], h[yh.flatten()==1,1], s=200, facecolors='none', edgecolors='red', marker='s', linewidths=2)

    if attempt_coord is not None:
        color = 'blue' if attempt_label == -1 else 'red'
        plt.scatter(attempt_coord[0], attempt_coord[1], s=250, facecolors='none', edgecolors=color, marker='s', linewidths=3)
    
    if mis_idx is not None:
        plt.scatter(X[mis_idx, 0], X[mis_idx, 1], color='black', marker='x', s=200, linewidths=3, label="Misclassified")

    plt.legend(loc='upper left')
    plt.grid(True)
    
    filename = f"{filename_base}_{FIG_COUNT:02d}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  💾 Figure saved as '{filename}'")
    plt.show()

# ============================================================
# Data Preparation
# ============================================================
np.random.seed(42)
n_samples = 30
X_left = np.random.randn(n_samples, 2) * 0.8 + np.array([-1, 0])
X_right = np.random.randn(n_samples, 2) * 0.8 + np.array([1, 0])
X = np.vstack((X_left, X_right))
y = np.hstack((-1 * np.ones(n_samples), np.ones(n_samples)))

class_weights = {-1: 1.0, 1.0: 1.0}

# ============================================================
# Run BSVM with Linear Kernel and Plot Stages
# ============================================================

parameters = dict(C=10, kernel='linear', class_weight=class_weights)
model = SVC(**parameters)

Xnew, ynew, h, yh = Initialsolution(X, y, model)

plot_decision_boundary(model, np.vstack([Xnew, h]), np.hstack([ynew, yh.flatten()]),
                       filename_base="initial_solution", h=h, yh=yh)

masterproblem(Xnew, ynew, X, y, h, yh, model, class_weights)
plot_decision_boundary(model, X, y, filename_base="final_model")

print("\n✅ BSVM with Linear kernel visualization complete. Figures saved to current directory.")