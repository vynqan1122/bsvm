# -*- coding: utf-8 -*-
"""
Verbose-traceable BSVM with Sigmoid Kernel - Manuscript Figure Generator (Zipped PDFs)
Now includes:
 • Displays plots interactively during execution.
 • Removes legends from all generated plots.
 • Saves all plots as sequentially numbered, high-resolution PDF files.
 • Automatically creates a zip archive of all figures.
 • Cleans up temporary files after execution.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import warnings
import os
import zipfile
import shutil

warnings.filterwarnings("ignore")

# --- Global Settings for Manuscript-Ready Plots ---
plt.rcParams.update({'font.size': 12})
FIG_COUNT = 0
FIGURE_DIR = "manuscript_figures_temp" # Temporary directory for PDFs
ZIP_FILENAME = "manuscript_figures.zip"

# ============================================================
# Helper Functions (Simplified for Non-Linear Kernels)
# ============================================================

def extend_samples(Xnew, ynew, X, y, h, yh, pipeline, weights=None):
    print("\n🔹 Entering extend_samples()")
    print(f"  Current training set size: {len(Xnew)}")
    print(f"  Remaining candidate pool size: {len(h)}")

    s = len(h)

    pipeline.set_params(class_weight={-1: 1.0, 1.0: 1.0})
    pipeline.fit(Xnew, ynew)
    
    print(f"  ✅ Model fitted on current Xnew/ynew")
    
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

            print(f"  ✅ Model re-fitted with {len(Xnew_temp)} samples")
            
            y_pred_Xnew = pipeline.predict(Xnew_temp)
            mis_idx = np.where(y_pred_Xnew != ynew_temp)[0]
            misclassified_flag = len(mis_idx) > 0
            print(f"  Misclassified? {misclassified_flag}")

            if misclassified_flag:
                print(f"  ❌ Adding sample ({candidate_coord[0]:.3f}, {candidate_coord[1]:.3f}) caused misclassification — rejecting.")
                plot_decision_boundary(pipeline, Xnew_temp, ynew_temp,
                                       filename_base="misclassification",
                                       mis_idx=mis_idx)
                pipeline.fit(Xnew, ynew)
                del b[idx]
                continue

            print(f"  ✅ Sample accepted ({candidate_coord[0]:.3f}, {candidate_coord[1]:.3f})")
            
            Xnew = Xnew_temp
            ynew = ynew_temp
            h = np.delete(h, idx, axis=0)
            yh = np.delete(yh, idx, axis=0)
            
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
    global FIG_COUNT, FIGURE_DIR
    FIG_COUNT += 1
    
    plt.figure(figsize=(4.5, 4))
    
    # Plot data points without labels
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color="blue", alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", alpha=0.6)

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
                   facecolors='none', edgecolors='k', linewidths=2)

    if h is not None and yh is not None:
        plt.scatter(h[yh.flatten()==-1,0], h[yh.flatten()==-1,1], s=200, facecolors='none', edgecolors='blue', marker='s', linewidths=2)
        plt.scatter(h[yh.flatten()==1,0], h[yh.flatten()==1,1], s=200, facecolors='none', edgecolors='red', marker='s', linewidths=2)

    if attempt_coord is not None:
        color = 'blue' if attempt_label == -1 else 'red'
        plt.scatter(attempt_coord[0], attempt_coord[1], s=250, facecolors='none', edgecolors=color, marker='s', linewidths=3)
    
    if mis_idx is not None:
        plt.scatter(X[mis_idx, 0], X[mis_idx, 1], color='black', marker='x', s=200, linewidths=3)

    # REMOVED: The legend is no longer displayed on the plot
    # plt.legend(loc='upper left')
    
    plt.grid(True)
    
    # Save the figure as PDF to the temporary directory
    filename = f"{filename_base}_{FIG_COUNT:02d}.pdf"
    filepath = os.path.join(FIGURE_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"  💾 Figure saved as '{filepath}'")
    
    # ADDED: Show the plot interactively
    plt.show()
    
    # Close the figure to free up memory
    plt.close(plt.gcf())

# ============================================================
# Main Execution Logic
# ============================================================

def main():
    if os.path.exists(FIGURE_DIR):
        shutil.rmtree(FIGURE_DIR)
    os.makedirs(FIGURE_DIR)

    try:
        np.random.seed(42)
        n_samples = 30
        X_left = np.random.randn(n_samples, 2) * 0.8 + np.array([-1, 0])
        X_right = np.random.randn(n_samples, 2) * 0.8 + np.array([1, 0])
        X = np.vstack((X_left, X_right))
        y = np.hstack((-1 * np.ones(n_samples), np.ones(n_samples)))
        class_weights = {-1: 1.0, 1.0: 1.0}

        # --- Using Sigmoid Kernel ---
        parameters = dict(C=10, kernel='sigmoid', gamma='scale', coef0=0, class_weight=class_weights)
        model = SVC(**parameters)

        Xnew, ynew, h, yh = Initialsolution(X, y, model)

        plot_decision_boundary(model, np.vstack([Xnew, h]), np.hstack([ynew, yh.flatten()]),
                               filename_base="initial_solution", h=h, yh=yh)

        masterproblem(Xnew, ynew, X, y, h, yh, model, class_weights)
        plot_decision_boundary(model, X, y, filename_base="final_model")

        print("\n✅ BSVM visualization complete. Zipping figures...")

    finally:
        print(f"Creating zip file: {ZIP_FILENAME}")
        with zipfile.ZipFile(ZIP_FILENAME, 'w') as zipf:
            for root, dirs, files in os.walk(FIGURE_DIR):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)
        
        print(f"Cleaning up temporary directory: {FIGURE_DIR}")
        shutil.rmtree(FIGURE_DIR)
        print("Done. All figures are in", ZIP_FILENAME)


if __name__ == "__main__":
    main()