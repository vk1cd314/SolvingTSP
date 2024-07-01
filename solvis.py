import lkh
import matplotlib.pyplot as plt
import math

def load_tsp_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def solve_tsp(problem_str, solver_path='LKH', max_trials=10000, runs=20):
    problem = lkh.LKHProblem.parse(problem_str)
    tour = lkh.solve(solver_path, problem=problem, max_trials=max_trials, runs=runs)
    return tour, problem.node_coords

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def parse_tsp_solution(tour, nodes):
    node_coords = nodes
    solution_coords = [node_coords[node] for node in tour]
    ans = 0
    n = len(solution_coords)
    for i in range(len(solution_coords)):
        ans += euclidean_distance(solution_coords[i], solution_coords[(i + 1) % n])
    # print(ans)
    return solution_coords, ans

def plot_tsp_solution(solution_coords, filename):
    x_coords, y_coords = zip(*solution_coords)
    plt.figure(figsize=(10, 10))
    plt.plot(x_coords, y_coords, 'bo-')
    plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 'bo-')  # Complete the loop
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('TSP Solution')
    plt.grid(True)
    filename = filename.replace('graphs/', '')
    print(filename)
    plt.savefig("images/" + filename.split('.')[0] + "_solution")
    # plt.show()

def vis_res(filename):
    tsp_file = filename  # Path to your .tsp file
    print(f"Solving {tsp_file}")
    solver_path = '/home/muhnay/Desktop/thesisstuff/SolvingTSP/LKH'  # Ensure LKH is in your PATH or provide the full path to the LKH executable

    problem_str = load_tsp_file(tsp_file)
    tour, nodes = solve_tsp(problem_str, solver_path)
    tour = tour[0]
    # print(tour)
    # print(nodes)

    solution_coords, ans = parse_tsp_solution(tour, nodes)
    plot_tsp_solution(solution_coords, filename)
    return ans
