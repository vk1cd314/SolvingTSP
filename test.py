import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

def read_data(filename):
    data = np.loadtxt(filename)
    # if filename != 'tsp200.res':
    #     for smol in data:
    #         tmp = copy.copy(smol[0])
    #         smol[0] = copy.copy(smol[1])
    #         smol[1] = tmp
    return data

def compute_optimality_gap(solutions, optimal_solution):
    return (solutions - optimal_solution) / optimal_solution * 100

files = ['tsp20.res', 'tsp50.res', 'tsp100.res', 'tsp200.res', 'tsp300.res', 'tsp400.res', 'tsp500.res']
data = [read_data(file) for file in files]

dense_gaps = []
sparse_gaps = []
times = []

for dataset in data:
    dense_sol = dataset[:, 0]
    sparse_sol = dataset[:, 1]
    time_for_spars = dataset[:, 2]
    optimal_sol = dataset[:, 3]
    
    dense_gap = compute_optimality_gap(dense_sol, optimal_sol)
    sparse_gap = compute_optimality_gap(sparse_sol, optimal_sol)
    
    dense_gaps.append(np.mean(dense_gap))
    sparse_gaps.append(np.mean(sparse_gap))
    times.append(np.mean(time_for_spars))

problem_sizes = [20, 50, 100, 200]

plt.figure(figsize=(10, 5))
plt.plot(problem_sizes, dense_gaps, label='Dense Graph Optimality Gap', marker='o')
plt.plot(problem_sizes, sparse_gaps, label='Sparse Graph Optimality Gap', marker='s')
plt.xlabel('Problem Size')
plt.ylabel('Optimality Gap (%)')
plt.title('Comparison of Optimality Gaps')
plt.legend()
plt.grid(True)
plt.savefig("results/optimality_gaps.pgf")

plt.figure(figsize=(10, 5))
plt.plot(problem_sizes, times, label='Time for Sparsification', marker='^')
plt.xlabel('Problem Size')
plt.ylabel('Time (seconds)')
plt.title('Sparsification Time')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("results/sparsification_time.pgf")
