import lkh
import matplotlib
import matplotlib.pyplot as plt
import math
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def load_tsp_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def solve_tsp(problem_str, solver_path='LKH', max_trials=100, runs=10):
    problem = lkh.LKHProblem.parse(problem_str)
    tour = lkh.solve(solver_path, problem=problem, max_trials=max_trials, runs=runs)
    return tour, problem.node_coords

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def parse_tsp_solution(tour, nodes, node_c=None):
    node_coords = nodes
    if len(nodes) == 0:
        node_coords = node_c
    solution_coords = [node_coords[node] for node in tour]
    ans = 0
    n = len(solution_coords)
    for i in range(len(solution_coords)):
        ans += euclidean_distance(solution_coords[i], solution_coords[(i + 1) % n])
    return solution_coords, ans

def plot_tsp_solution(solution_coords, filename):
    x_coords, y_coords = zip(*solution_coords)
    plt.figure(figsize=(2.2, 2.2))
    plt.plot(x_coords, y_coords, 'bo-')
    plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 'bo-')  # Complete the loop
    plt.title('TSP Solution')
    plt.axis('off')
    filename = filename.replace('graphs/', '')
    plt.savefig("images/" + filename.split('.')[0] + "_solution.pgf")

def vis_res(filename, node_coords=None):
    tsp_file = filename  
    print(f"Solving {tsp_file}")
    solver_path = './LKH' 

    problem_str = load_tsp_file(tsp_file)
    tour, nodes = solve_tsp(problem_str, solver_path)
    tour = tour[0]

    solution_coords, ans = parse_tsp_solution(tour, nodes, node_coords)
    # plot_tsp_solution(solution_coords, filename)
    return ans
