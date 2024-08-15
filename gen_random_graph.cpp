#include <bits/stdc++.h>

using namespace std;

#define x first 
#define y second
#define pt pair<int, int>

int euclidean_distance(pt A, pt B) {
  return sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
}

int main(int argc, char **argv) {
  int n = atoi(argv[1]), seed = atoi(argv[2]);
  srand(seed);
  int grid_size = max(5 * n, 100);
  vector <pt> pts(n);
  for (int i = 0; i < n; ++i) pts[i].x = rand() % grid_size + 1, pts[i].y = rand() % grid_size + 1;
  cout << n << ' ' << n * (n - 1) / 2 << '\n';
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
       cout << i << ' ' << j << ' ' << euclidean_distance(pts[i], pts[j]) << '\n';
    }
  }
}
