#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <iomanip>

struct Edge {
    int u, v;
    double weight;
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream inputFile(argv[1]);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << argv[1] << std::endl;
        return 1;
    }

    int numNodes, numEdges;
    inputFile >> numNodes >> numEdges;

    std::vector<Edge> edges(numEdges);
    double maxWeight = std::numeric_limits<double>::min();

    for (int i = 0; i < numEdges; ++i) {
        int u, v;
        double weight;
        inputFile >> u >> v >> weight;
        edges[i] = {u, v, weight};

        if (weight > maxWeight) {
            maxWeight = weight;
        }
    }

    inputFile.close();

    std::cout << numNodes << " " << numEdges << std::endl;
    std::cout << std::fixed << std::setprecision(6); // Format output to 6 decimal places
    for (const auto& edge : edges) {
        double normalizedWeight = edge.weight / (1+maxWeight);
        std::cout << edge.u << " " << edge.v << " " << normalizedWeight << std::endl;
    }

    return 0;
}

