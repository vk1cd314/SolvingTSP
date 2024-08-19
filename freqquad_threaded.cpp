#include <bits/stdc++.h>
#include <omp.h> // OpenMP header for parallelization

using namespace std;

const int N = 1e7, M = 1e3 + 10;
const int MX_REP = 1e8;
int edge_map[M][M];

int get_freq(int a, int b, int c, int d) {
  double ab = edge_map[a][b]; 
  double bc = edge_map[b][c];
  double cd = edge_map[c][d];
  double ad = edge_map[a][d];
  double ac = edge_map[a][c];
  double bd = edge_map[b][d];

  if (ab == N || bc == N || cd == N || ad == N) return -1;

  ab += rand() % N / (1.0 * N); 
  bc += rand() % N / (1.0 * N);
  cd += rand() % N / (1.0 * N);
  ad += rand() % N / (1.0 * N);
  ac += rand() % N / (1.0 * N);
  bd += rand() % N / (1.0 * N);

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
  int t_count = atoi(argv[2]);
  file.open(filename);
  int vertices, edges;
  file >> vertices >> edges;
  int v1, v2;
  int length;

  const int QUAD_NO = min(vertices, 120);

  vector<array<int, 3>> edge_list;
  while (file >> v1 >> v2 >> length) {
    if (v1 > v2) swap(v1, v2);
    edge_map[v1][v2] = length;
    edge_map[v2][v1] = length;
    edge_list.push_back({v1, v2, 0});
  }
  file.close();

  auto start = chrono::high_resolution_clock::now();

  for (int turn = 0; turn < t_count; ++turn) {
    bool ends = false;

    // Parallelize the loop over the edges using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < (int)edge_list.size(); ++i) {
      auto &[u, v, edge_freq_e] = edge_list[i];

      for (int j = 0; j < QUAD_NO; ++j) {
        int cnt = 0;
      st:
        cnt += 1;
        if (cnt > MX_REP) {
            ends = true;
            break;
        }
        int u1 = rand() % vertices;
        while (u1 == u || u1 == v)
          u1 = rand() % vertices;
        int v1 = rand() % vertices;
        while (v1 == u || v1 == v || u1 == v1)
          v1 = rand() % vertices;
        int freq = get_freq(u, v, u1, v1);
        if (freq == -1) goto st;
        edge_freq_e += freq;
      }
    }

    if (ends) {
      cerr << "broke at turn " << turn + 1 << '\n';
      break;
    }

    vector<array<int, 3>> new_edge_list;
    for (int i = 0; i < (int)edge_list.size(); ++i) {
      if (edge_list[i][2] >= 3 * QUAD_NO) {
        new_edge_list.push_back({edge_list[i][0], edge_list[i][1], 0});
      } else {
        edge_map[edge_list[i][0]][edge_list[i][1]] = N;
        edge_map[edge_list[i][1]][edge_list[i][0]] = N;
      }
    }

    edge_list = new_edge_list;

    // Optionally log the time taken for each turn
    auto end_turn = chrono::high_resolution_clock::now();
    auto duration_turn = chrono::duration_cast<chrono::microseconds>(end_turn - start);
    cerr << "Time need for turn " << turn + 1 << ": " << duration_turn.count() / 1000000.0 << " seconds\n";
    cerr << new_edge_list.size() << '\n';
  }

  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
  cerr << "Time taken total: " << duration.count() / 1000000.0 << " seconds" << '\n';
  cout << vertices << " " << edge_list.size() << '\n';

  for (int i = 0; i < (int)edge_list.size(); i++) {
    int u = edge_list[i][0];
    int v = edge_list[i][1];
    int w = edge_map[u][v];
    cout << u << " " << v << " " << w << '\n';
  }

  return 0;
}

