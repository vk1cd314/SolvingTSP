#include <bits/stdc++.h>

using namespace std;
const int N = 1e7;

map<pair<int, int>, double> get_freq(int a, int b, int c, int d,
                                     map<pair<int, int>, double> &edgeList) {
  map<pair<int, int>, double> freq_dict;

  double ab = edgeList[{a, b}] + rand() % N / (1.0 * N);
  double bc = edgeList[{b, c}] + rand() % N / (1.0 * N);
  double cd = edgeList[{c, d}] + rand() % N / (1.0 * N);
  double ad = edgeList[{a, d}] + rand() % N / (1.0 * N);
  double ac = edgeList[{a, c}] + rand() % N / (1.0 * N);
  double bd = edgeList[{b, d}] + rand() % N / (1.0 * N);

  // Fill the frequency dictionary based on the conditions
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
  double length;

  map<pair<int, int>, double> edge_map;
  while (file >> v1 >> v2 >> length) {
    // if (v1 > v2) swap(v1, v2);
    edge_map[{v1, v2}] = length;
    edge_map[{v2, v1}] = length;
  }
  file.close();
  map<pair<int, int>, double> edge_freq;
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
      if (checked[{u, v, min(u1, v1), max(u1, v1)}] == 1)
        continue;
      checked[{u, v, min(u1, v1), max(u1, v1)}] = 1;
      auto freq_dict = get_freq(u, v, u1, v1, edge_map);
      edge_freq_e += freq_dict[{u, v}];
    }
    edge_freq[{u, v}] = 1.0 * edge_freq_e / vertices;
  }
  vector<tuple<int, int, double>> sorted_edges;
  for (auto [edge, len] : edge_freq)
    sorted_edges.push_back({get<0>(edge), get<1>(edge), len});
  sort(sorted_edges.begin(), sorted_edges.end(),
       [](tuple<int, int, double> a, tuple<int, int, double> b) {
         return get<2>(a) > get<2>(b);
       });
  cout << vertices << " " << (2 * edges + 2) / 3 << endl;
  vector<tuple<int, int, double>> to_write;
  for (int i = 0; i < (2 * edges + 2) / 3; i++) {
    int u = get<0>(sorted_edges[i]);
    int v = get<1>(sorted_edges[i]);
    double w = edge_map[{u, v}];
    // cerr << "what? " << w << '\n';
    to_write.push_back({min(u, v), max(u, v), w});
    // cout << u << " " << v << " " << w << endl;
  }
  sort(to_write.begin(), to_write.end());
  // cout << fixed << setprecision(10);
  for (int i = 0; i < to_write.size(); i++) {
    cout << get<0>(to_write[i]) << " " << get<1>(to_write[i]) << " "
         << get<2>(to_write[i]) << endl;
  }
  cerr << "Final size: " << to_write.size() << '\n';
  return 0;
}
