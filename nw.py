import lkh

def load_tsp_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def solve_tsp(problem_str, solver_path='./LKH', max_trials=100, runs=10):
    problem = lkh.LKHProblem.parse(problem_str)
    tour = lkh.solve(solver_path, problem=problem, max_trials=max_trials, runs=runs)
    return tour, problem.node_coords

print(solve_tsp(load_tsp_file("graphs/original_graph.tsp")))
print(solve_tsp(load_tsp_file("graphs/sparsified_graph.tsp")))
