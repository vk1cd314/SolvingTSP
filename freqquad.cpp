#include <bits/stdc++.h>
#include <chrono>

using namespace std;
const int N = 1e7;

map<pair<int, int>, int> get_freq(int a, int b, int c, int d,
                                     map<pair<int, int>, int> &edgeList) {
  map<pair<int, int>, int> freq_dict;

  int ab = edgeList[{a, b}] + rand() % N / (1.0 * N);
  int bc = edgeList[{b, c}] + rand() % N / (1.0 * N);
  int cd = edgeList[{c, d}] + rand() % N / (1.0 * N);
  int ad = edgeList[{a, d}] + rand() % N / (1.0 * N);
  int ac = edgeList[{a, c}] + rand() % N / (1.0 * N);
  int bd = edgeList[{b, d}] + rand() % N / (1.0 * N);

  if (ab + cd < ac + bd && ac + bd < ad + bc) {
    freq_dict[{a, b}] = 5;
    freq_dict[{b, c}] = 1;
    freq_dict[{c, d}] = 5;
    freq_dict[{a, d}] = 1;
    freq_dict[{a, c}] = 3;
    freq_dict[{b, d}] = 3;
  } else if (ab + cd < ad + bc && ad + bc < ac + bd) {
    freq_dict[{a, b}] = 5;
    freq_dict[{b, c}] = 3;
    freq_dict[{c, d}] = 5;
    freq_dict[{a, d}] = 3;
    freq_dict[{a, c}] = 1;
    freq_dict[{b, d}] = 1;
  } else if (ac + bd < ab + cd && ab + cd < ad + bc) {
    freq_dict[{a, b}] = 3;
    freq_dict[{b, c}] = 1;
    freq_dict[{c, d}] = 3;
    freq_dict[{a, d}] = 1;
    freq_dict[{a, c}] = 5;
    freq_dict[{b, d}] = 5;
  } else if (ac + bd < ad + bc && ad + bc < ab + cd) {
    freq_dict[{a, b}] = 1;
    freq_dict[{b, c}] = 3;
    freq_dict[{c, d}] = 1;
    freq_dict[{a, d}] = 3;
    freq_dict[{a, c}] = 5;
    freq_dict[{b, d}] = 5;
  } else if (ad + bc < ab + cd && ab + cd < ac + bd) {
    freq_dict[{a, b}] = 3;
    freq_dict[{b, c}] = 5;
    freq_dict[{c, d}] = 3;
    freq_dict[{a, d}] = 5;
    freq_dict[{a, c}] = 1;
    freq_dict[{b, d}] = 1;
  } else if (ad + bc < ac + bd && ac + bd < ab + cd) {
    freq_dict[{a, b}] = 1;
    freq_dict[{b, c}] = 5;
    freq_dict[{c, d}] = 1;
    freq_dict[{a, d}] = 5;
    freq_dict[{a, c}] = 3;
    freq_dict[{b, d}] = 3;
  }

  return freq_dict;
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
  while (file >> v1 >> v2 >> length) {
    edge_map[{v1, v2}] = length;
    edge_map[{v2, v1}] = length;
  }
  file.close();
  auto start = std::chrono::high_resolution_clock::now();
  map<pair<int, int>, int> edge_freq;
  map<pair<int, int>, bool> done;
  for (auto [edge, len] : edge_map) {
    int edge_freq_e = 0;
    auto [u, v] = edge;
    if (done[{min(u, v), max(u, v)}] == 1)
      continue;
    done[{min(u, v), max(u, v)}] = 1;
    map<tuple<int, int, int, int>, bool> checked;
    vector<pair<int, int>> freq_quads;
    for (int i = 0; i < vertices; ++i) {
      int u1 = rand() % vertices;
      while (u1 == u || u1 == v)
        u1 = rand() % vertices;
      int v1 = rand() % vertices;
      while (v1 == u || v1 == v || u1 == v1)
        v1 = rand() % vertices;
      freq_quads.push_back({u1, v1});
    }
    for (int i = 0; i < vertices; i++) {
      auto [u1, v1] = freq_quads[i];
      auto freq_dict = get_freq(u, v, u1, v1, edge_map);
      edge_freq_e += freq_dict[{u, v}];
    }
    edge_freq[{u, v}] = 1.0 * edge_freq_e / vertices;
  }
  vector<tuple<int, int, int>> sorted_edges;
  for (auto [edge, len] : edge_freq)
    sorted_edges.push_back({get<0>(edge), get<1>(edge), len});
  sort(sorted_edges.begin(), sorted_edges.end(),
       [](tuple<int, int, int> a, tuple<int, int, int> b) {
         return get<2>(a) > get<2>(b);
       });
  auto ed = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(ed - start);
  cerr << "Time taken: " << duration.count() / 1000000.0 << " seconds" << '\n';
  cout << vertices << " " << (2 * edges + 2) / 3 << '\n';
  vector<tuple<int, int, int>> to_write;
  for (int i = 0; i < (2 * edges + 2) / 3; i++) {
    int u = get<0>(sorted_edges[i]);
    int v = get<1>(sorted_edges[i]);
    int w = edge_map[{u, v}];
    cout << u << " " << v << " " << w << '\n';
  }
  return 0;
}
