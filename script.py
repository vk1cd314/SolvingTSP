import subprocess
from tqdm import tqdm

numbers = [20, 50, 100, 200, 300, 400, 500]

for number in numbers:
    print(f"Started Batch {number}")
    with tqdm(total=10, desc=f"Batch {number}") as pbar:
        for i in range(10):
            subprocess.run(['python', 'sparse.py', str(number)])
            
            result = subprocess.run(['./concorde-bin', '-o', 'output.sol', 'graphs/original_graph.tsp'], 
                                    capture_output=True, text=True)
            with open(f'tsp{number}.res', 'a') as f:
                for line in result.stdout.splitlines():
                    if "Optimal Solution:" in line:
                        f.write(line.replace('Optimal Solution:', '') + '\n')
            
            pbar.update(1)
    print(f"Finished Batch {number}")
