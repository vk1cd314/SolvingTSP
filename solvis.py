import lkh
import matplotlib.pyplot as plt

def load_tsp_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def solve_tsp(problem_str, solver_path='LKH', max_trials=100, runs=10):
    problem = lkh.LKHProblem.parse(problem_str)
    tour = lkh.solve(solver_path, problem=problem, max_trials=max_trials, runs=runs)
    return tour, problem.nodes

def parse_tsp_solution(tour, nodes):
    node_coords = {node.id: (node.x, node.y) for node in nodes}
    solution_coords = [node_coords[node] for node in tour]
    return solution_coords

def plot_tsp_solution(solution_coords):
    x_coords, y_coords = zip(*solution_coords)
    plt.figure(figsize=(10, 10))
    plt.plot(x_coords, y_coords, 'bo-')
    plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 'bo-')  # Complete the loop
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('TSP Solution')
    plt.grid(True)
    plt.show()

def vis_res(filename):
    tsp_file = filename  # Path to your .tsp file
    solver_path = 'LKH'  # Ensure LKH is in your PATH or provide the full path to the LKH executable

    problem_str = load_tsp_file(tsp_file)
    tour, nodes = solve_tsp(problem_str, solver_path)

    solution_coords = parse_tsp_solution(tour, nodes)
    plot_tsp_solution(solution_coords)

