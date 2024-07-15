#include <stdio.h>
#include <stdlib.h>

// Struct to represent an edge
typedef struct {
    int u;
    int v;
    int length;
    int frequency;
} Edge;

// Function to compare edges based on frequency (for sorting)
int compare(const void* a, const void* b) {
    Edge* edgeA = (Edge*)a;
    Edge* edgeB = (Edge*)b;
    return edgeB->frequency - edgeA->frequency;
}

int main() {
    FILE* file = fopen("input.txt", "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    int num_vertices, num_edges;
    fscanf(file, "%d %d", &num_vertices, &num_edges);

    Edge* edges = (Edge*)malloc(num_edges * sizeof(Edge));
    for (int i = 0; i < num_edges; ++i) {
        fscanf(file, "%d %d %d", &edges[i].u, &edges[i].v, &edges[i].length);
        edges[i].frequency = 0;  // Initialize frequency
    }
    fclose(file);

    // Calculate frequencies based on the given rules
    for (int i = 0; i < num_edges; ++i) {
        int u1 = edges[i].u;
        int v1 = edges[i].v;

        for (int j = 0; j < num_edges; ++j) {
            if (j == i) continue;
            int u2 = edges[j].u;
            int v2 = edges[j].v;

            if (u2 == u1 || v2 == u1) {
                int vX = (u2 == u1) ? v2 : u2;

                for (int k = 0; k < num_edges; ++k) {
                    if (k == i || k == j) continue;
                    int u3 = edges[k].u;
                    int v3 = edges[k].v;

                    if (u3 == v1 || v3 == v1) {
                        int vY = (u3 == v1) ? v3 : u3;

                        for (int l = 0; l < num_edges; ++l) {
                            if (l == i || l == j || l == k) continue;
                            if ((edges[l].u == vX && edges[l].v == vY) || (edges[l].u == vY && edges[l].v == vX)) {
                                int ab = edges[i].length;
                                int ac = edges[j].length;
                                int bd = edges[k].length;
                                int cd = edges[l].length;

                                // Update frequencies based on the given conditions
                                if (ab + cd < ac + bd) {
                                    edges[i].frequency += 5;
                                    edges[j].frequency += 1;
                                    edges[k].frequency += 5;
                                    edges[l].frequency += 1;
                                } else if (ac + bd < ab + cd) {
                                    edges[i].frequency += 1;
                                    edges[j].frequency += 5;
                                    edges[k].frequency += 1;
                                    edges[l].frequency += 5;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort edges by frequency in descending order
    qsort(edges, num_edges, sizeof(Edge), compare);

    // Print the edges with their frequencies
    for (int i = 0; i < num_edges; ++i) {
        printf("%d %d %d %d\n", edges[i].u, edges[i].v, edges[i].length, edges[i].frequency);
    }

    free(edges);
    return 0;
}
