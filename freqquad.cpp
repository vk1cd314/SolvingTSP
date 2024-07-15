#include <iostream>
#include <vector>
#include <tuple>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/hash_policy.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <set>
#include <fstream>

using namespace std;
using namespace __gnu_pbds;

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2; // XORing the two hash values
    }
};
gp_hash_table<pair<int, int>, double, pair_hash> getFreq(int a, int b, int c, int d, gp_hash_table<pair<int, int>, double, pair_hash> edgeList) {
    gp_hash_table<pair<int, int>, double, pair_hash> freq_dict;
    
    // Calculate the distances with random perturbation

    double ab = edgeList[{a, b}];
    double bc = edgeList[{b, c}];
    double cd = edgeList[{c, d}];
    double ad = edgeList[{a, d}];
    double ac = edgeList[{a, c}];
    double bd = edgeList[{b, d}];

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
        freq_dict[{a, a}] = 3;
        freq_dict[{b, c}] = 1;
        freq_dict[{c, d}] = 3;
        freq_dict[{a, d}] = 1;
        freq_dict[{a, c}] = 5;
        freq_dict[{b, d}] = 5;
    } else if (ac + bd < ad + bc && ad + bc < ab + cd) {
        freq_dict[{a, a}] = 1;
        freq_dict[{b, c}] = 3;
        freq_dict[{c, d}] = 1;
        freq_dict[{a, d}] = 3;
        freq_dict[{a, c}] = 5;
        freq_dict[{b, d}] = 5;
    } else if (ad + bc < ab + cd && ab + cd < ac + bd) {
        freq_dict[{a, a}] = 3;
        freq_dict[{b, c}] = 5;
        freq_dict[{c, d}] = 3;
        freq_dict[{a, d}] = 5;
        freq_dict[{a, c}] = 1;
        freq_dict[{b, d}] = 1;
    } else if (ad + bc < ac + bd && ac + bd < ab + cd) {
        freq_dict[{a, a}] = 1;
        freq_dict[{b, c}] = 5;
        freq_dict[{c, d}] = 1;
        freq_dict[{a, d}] = 5;
        freq_dict[{a, c}] = 3;
        freq_dict[{b, d}] = 3;
    }

    return freq_dict;
}

int main(){
    srand(time(0));
    ifstream file;
    file.open("input.txt");
    int vertices, edges;
    file >> vertices >> edges;
    int v1, v2, length;

    gp_hash_table <pair<int, int>, double, pair_hash> edgeMap;
    while(file >> v1 >> v2 >> length){
        if (v1 > v2) swap(v1, v2);
        edgeMap[{v1, v2}] = length;
    }
    file.close();
    gp_hash_table<pair <int, int>, double, pair_hash > edge_freq;
    for (auto [edge, len]: edgeMap) {
        int edge_freq_e = 0;
        auto [u, v] = edge;
        gp_hash_table <pair<int, int>, bool, pair_hash> checked;
        for(int i=0; i<vertices; i++){
            int u1 = rand()%vertices;
            while(u1==u|| u1==v){
                u1 = rand()%vertices;
            }
            int v1 = rand()%vertices;
            while(v1==u|| v1==v||u1==v1){
                v1 = rand()%vertices;
            }
            if(checked[{min(u1, v1), max(u1, v1)}]==1){
                continue;
            }
            checked[{min(u1, v1), max(u1, v1)}] = 1;
            auto freq_dict = getFreq(u, v, u1, v1, edgeMap);
            edge_freq_e += freq_dict[{u, v}];
        }
        edge_freq[{u, v}] = 1.0 * edge_freq_e/vertices;
    }
    vector<tuple<int, int, double>> sorted_edges;
    for(auto [edge, len]: edge_freq){
        sorted_edges.push_back({get<0>(edge), get<1>(edge), len});
    }
    sort(sorted_edges.begin(), sorted_edges.end(), [](tuple<int, int, double> a, tuple<int, int, double> b){
        return get<2>(a) > get<2>(b);
    });
    cout << vertices << " " << (2*edges+2)/3 << endl;
    for (int i=0; i< (2*edges+2)/3; i++){
        int u = get<0>(sorted_edges[i]);
        int v = get<1>(sorted_edges[i]);
        double w = edgeMap[{u,v}];
        cout << u << " " << v << " " << w << endl;
    }
    return 0;
}