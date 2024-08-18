#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace std;
using namespace __gnu_pbds;

const int N = 1e7;

pair <int, int> smol(int a, int b) {
  if (a > b) swap(a, b);
  return {a, b};
}

int get_freq(int a, int b, int c, int d,
              map<pair<int, int>, int> &edgeList) {

  double ab = edgeList[smol(a, b)] + rand() % N / (1.0 * N);
  double bc = edgeList[smol(b, c)] + rand() % N / (1.0 * N);
  double cd = edgeList[smol(c, d)] + rand() % N / (1.0 * N);
  double ad = edgeList[smol(a, d)] + rand() % N / (1.0 * N);
  double ac = edgeList[smol(a, c)] + rand() % N / (1.0 * N);
  double bd = edgeList[smol(b, d)] + rand() % N / (1.0 * N);

  if (ab + cd < ac + bd && ac + bd < ad + bc) return 5;
  if (ab + cd < ad + bc && ad + bc < ac + bd) return 5;
  if (ac + bd < ab + cd && ab + cd < ad + bc) return 3;
  if (ac + bd < ad + bc && ad + bc < ab + cd) return 1;
  if (ad + bc < ab + cd && ab + cd < ac + bd) return 3;
  if (ad + bc < ac + bd && ac + bd < ab + cd) return 1;

  return 0;
}

int main(int argc, char **argv) {
  srand(time(0));
  ifstream file;
  char *filename = argv[1];
  file.open(filename);
  int vertices, edges;
  file >> vertices >> edges;
  int v1, v2;
  int length;

  map<pair<int, int>, int> edge_map;
  vector <array <int, 3>> edge_list;
  while (file >> v1 >> v2 >> length) {
    if (v1 > v2) swap(v1, v2);
    edge_map[{v1, v2}] = length;
    edge_list.push_back({v1, v2, 0});
  }
  file.close();

  auto start = chrono::high_resolution_clock::now();
  
  for (int i = 0; i < (int) edge_list.size(); ++i) {
    int edge_freq_e = 0;
    auto &[u, v, avg_freq] = edge_list[i];

    for (int i = 0; i < vertices; i++) {
      int u1 = rand() % vertices;
      while (u1 == u || u1 == v)
        u1 = rand() % vertices;
      int v1 = rand() % vertices;
      while (v1 == u || v1 == v || u1 == v1)
        v1 = rand() % vertices;
      edge_freq_e += get_freq(u, v, u1, v1, edge_map);
    }

    avg_freq = 1.0 * edge_freq_e / vertices;
  }

  sort(edge_list.begin(), edge_list.end(),
       [](array<int, 3> a, array<int, 3> b) {
         return a[2] > b[2];
       });
  
  auto ed = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(ed - start);
  cerr << "Time taken: " << duration.count() / 1000000.0 << " seconds" << '\n';
  cout << vertices << " " << (2 * edges + 2) / 3 << '\n';
 
  for (int i = 0; i < (2 * edges + 2) / 3; i++) {
    int u = edge_list[i][0]; 
    int v = edge_list[i][1];
    int w = edge_map[{u, v}];
    cout << u << " " << v << " " << w << '\n';
  }
}
