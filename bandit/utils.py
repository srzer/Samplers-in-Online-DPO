import numpy as np
import pdb
import random

from collections import defaultdict

def assemble_trajectories(dataset, max_length):
    n = 0
    mapping = {}
    inverse_mapping = []

    def query_idx(obs):
        nonlocal n, mapping
        key = tuple(obs)
        if key not in mapping:
            mapping[key] = n
            inverse_mapping.append(key)
            n += 1
        return mapping[key]

    edges, degree = defaultdict(list), defaultdict(int)
    m = dataset["observations"].shape[0]
    for i in range(m):
        u = query_idx(dataset["observations"][i])
        v = query_idx(dataset["next_observations"][i])
        edges[u].append((v, i))
        degree[v] += 1

    queue = []
    for u in range(n):
        if degree[u] == 0:
            queue.append(u)
        random.shuffle(edges[u])

    trajectories = []
    while len(queue):
        u = queue.pop(0)
        while len(edges[u]):
            cur_u = u
            trajectory = []
            while len(trajectory) < max_length and len(edges[cur_u]):
                v, i = edges[cur_u].pop()
                trajectory.append({
                    "s": inverse_mapping[cur_u],
                    "a": dataset["actions"][i],
                    "r": dataset["rewards"][i],
                })

                degree[v] -= 1
                if degree[v] == 0:
                    queue.append(v)
                cur_u = v
            
            trajectories.append(trajectory)
    
    for u in range(n):
        assert len(edges[u]) == 0, "The graph is not a DAG"
    
    return trajectories
