#include <bits/stdc++.h>
#include <cstdlib>

using namespace std;

int main(int argc, char **argv) {
    int seed = atoi(argv[2]);
    int n = atoi(argv[1]);
    int BIG_NUMBER = 1e4;
    vector <tuple <int, int, int>> edges;
    map <pair <int, int>, int> have;
    for (int i = 0; i < n; ++i) {
        edges.push_back({i, (i + 1) % n, rand() % BIG_NUMBER + 1});
        have[{i, (i + 1) % n}] = 1;
        have[{(i + 1) % n, i}] = 1;
    }
    int EXTRAS = 3 * n;
    for (int i = 0; i < EXTRAS; ++i) {
        int u = rand() % n, v = rand() % n;
        while (u == v) v = rand() % n;
        if (have[{u, v}] or have[{v, u}]) {
            i -= 1;
            continue;
        }
        have[{u, v}] = 1;
        have[{v, u}] = 1;   
        edges.push_back({u, v, rand() % BIG_NUMBER + 1});
    }
    cout << n << ' ' << edges.size() << '\n';
    for (auto [u, v, w] : edges) {
        cout << u << ' ' << v << ' ' << w << '\n';
    }
}
