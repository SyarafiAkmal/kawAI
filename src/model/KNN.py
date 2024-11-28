import csv

with open('data/dummy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    array_2d = [row for row in reader]

def distance(query, row):
    dist = 0
    for i in range(len(query)):
        if query[i] != row[i]:
            dist += 1
    return dist

def KNN(k, query):
    dist_array = [[0, i]for i in range(len(array_2d) - 1)]
    for i in range(len(array_2d) - 1):
        dist_array[i][0] = distance(query, array_2d[i])
    
    sorted_dist = sorted(dist_array, key=lambda x: x[0])
    for i in range(k):
        print(f"D-{sorted_dist[i][1]}: {array_2d[sorted_dist[i][1]]}, distance: {sorted_dist[i][0]}")
    

# KNN(3, ['sunny', 'cool', 'high', 'true'])






