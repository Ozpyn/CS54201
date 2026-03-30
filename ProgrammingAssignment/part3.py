# Write a K-means clustering program, and test it out on some practical problem using data generated using random number generator or data read from a file.

import random
import math
import matplotlib.pyplot as plt

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def mean_point(points):
    if not points:
        return (0, 0)
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)

def generate_data(filename, num_points, clusters, spread):
    centers = [
        (random.uniform(-10, 10), random.uniform(-10, 10))
        for _ in range(clusters)
    ]

    with open(filename, "w") as f:
        for _ in range(num_points):
            cx, cy = random.choice(centers)
            x = random.gauss(cx, spread)
            y = random.gauss(cy, spread)
            f.write(f"{x},{y}\n")

    print(f"Data generated and saved to {filename}")

def read_points(filename):
    points = []
    with open(filename, "r") as f:
        for line in f:
            x, y = map(float, line.strip().split(","))
            points.append((x, y))
    return points

def generate_seed_points(SP, nc):
    xs = [p[0] for p in SP]
    ys = [p[1] for p in SP]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    size_x = (max_x - min_x) / nc
    size_y = (max_y - min_y) / nc

    nm = nc * nc
    density = len(SP) / nm

    Sh = []

    for i in range(nc):
        for j in range(nc):
            xlow = min_x + i * size_x
            xhigh = xlow + size_x
            ylow = min_y + j * size_y
            yhigh = ylow + size_y

            block_points = [
                p for p in SP
                if xlow <= p[0] < xhigh and ylow <= p[1] < yhigh
            ]

            if len(block_points) > density:
                xmid = (xlow + xhigh) / 2
                ymid = (ylow + yhigh) / 2
                Sh.append((xmid, ymid))

    if len(Sh) < nc:
        Sh = SP.copy()

    ST = random.sample(Sh, nc)

    radius = float("inf")
    for i in range(len(ST)):
        for j in range(len(ST)):
            if i != j:
                d = euclidean_distance(ST[i], ST[j])
                if d < 2 * radius:
                    radius = d / 2

    return ST, radius

def plot_clusters(clusters, centroids, outliers, iteration):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

    plt.figure(figsize=(7, 7))

    for i, cluster in enumerate(clusters):
        if cluster:
            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            plt.scatter(xs, ys, color=colors[i % len(colors)], label=f"Cluster {i+1}")

    for i, c in enumerate(centroids):
        plt.scatter(
            c[0], c[1],
            color='black',
            marker='X',
            s=200,
            edgecolors='white',
            linewidths=2,
            label="Centroids" if i == 0 else ""
        )

    if outliers:
        xs = [p[0] for p in outliers]
        ys = [p[1] for p in outliers]
        plt.scatter(xs, ys, color='black', marker='.', label="Outliers")

    plt.title(f"Iteration {iteration}")
    plt.legend()
    plt.grid(True)
    plt.pause(0.5)

def k_means_clustering(filename, nc, max_iter, max_shift):
    SP = read_points(filename)

    centroids, radius = generate_seed_points(SP, nc)

    plt.ion()

    count = 0
    stabilized = False

    while count < max_iter and not stabilized:
        clusters = [[] for _ in range(nc)]
        outliers = SP.copy()

        for i in range(nc):
            for p in SP:
                if euclidean_distance(centroids[i], p) < radius:
                    clusters[i].append(p)
                    if p in outliers:
                        outliers.remove(p)

        new_centroids = []
        for i in range(nc):
            c = mean_point(clusters[i])
            new_centroids.append(c)
            print(f"Cluster {i+1} centroid: {c}")

        print("Outliers:", outliers)

        plot_clusters(clusters, new_centroids, outliers, count)

        stabilized = True
        for i in range(nc):
            if euclidean_distance(centroids[i], new_centroids[i]) > max_shift:
                stabilized = False

        centroids = new_centroids
        count += 1

    plt.ioff()
    plt.show()

    print("\nFinal Clusters:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")

    print("Final Outliers:", outliers)

filename = "data.txt"

generate_data(filename, num_points=200, clusters=5, spread=0.8)

k_means_clustering(filename, nc=5, max_iter=15, max_shift=0.001)