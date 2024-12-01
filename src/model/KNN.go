package model

import (
	"fmt"
	"sort"
	"sync"
)

type KNearestNeighbor struct {
	dataTrainX [][]string
	dataTrainY [][]string
	dataTestX  [][]string
	dataTestY  [][]string
}

func (knn *KNearestNeighbor) distance(query []string, row []string) int {
	// var wg sync.WaitGroup
	// var mu sync.Mutex

	dist := 0
	// for i := range query {
	// 	wg.Add(1)
	// 	go func(i int) {
	// 		defer wg.Done()
	// 		if query[i] != row[i] {
	// 			mu.Lock()
	// 			dist++
	// 			mu.Unlock()
	// 		}
	// 	}(i)
	// }

	// wg.Wait()

	for i := range query {
		if query[i] != row[i] {
			dist++
		}
	}

	return dist
}


func (knn *KNearestNeighbor) predict(k int, query []string) string {
	distArray := make([][2]int, len(knn.dataTrainX))

	for i := range knn.dataTrainX {
		distArray[i] = [2]int{knn.distance(query, knn.dataTrainX[i]), i}
	}

	sort.Slice(distArray, func(i, j int) bool {
		if distArray[i][0] == distArray[j][0] {
			return distArray[i][1] < distArray[j][1]
		}
		return distArray[i][0] < distArray[j][0]
	})

	// fmt.Println(distArray)
	distArray = distArray[:k]

	result := make(map[string]int)
	for _, pair := range distArray {
		label := knn.dataTrainY[pair[1]][0]
		result[label]++
	}

	maxCount := 0
	mode := ""
	for label, count := range result {
		if count > maxCount {
			maxCount = count
			mode = label
		}
	}

	// fmt.Println(result)
	return mode
}

func (knn *KNearestNeighbor) score(k int) float64 {
	var wg sync.WaitGroup
	var mu sync.Mutex

	tp := 0

	for i, query := range knn.dataTestX {
		wg.Add(1)
		go func(i int, query []string) {
			defer wg.Done()

			prediction := knn.predict(k, query)
			if prediction == knn.dataTestY[i][0] {
				mu.Lock() // Lock tp variable
				tp++
				mu.Unlock() // Unlock after modifying tp
			}
			// fmt.Println(i)
			// fmt.Printf("Current Accuracy: %.2f\n", float64(tp)/float64(i+1))
		}(i, query)
	}
	wg.Wait()

	return float64(tp) / float64(len(knn.dataTestX))
}

func MainKNN(data [][]string) {

	var features [][]string
	var labels [][]string
	for _, row := range data {
		features = append(features, row[:len(row)-1])
		labels = append(labels, []string{row[len(row)-1]})
	}

	// trainX := features
	// testX := features
	// trainY := labels
	// testY := labels
	trainX, testX := splitDataset(features, 0.8)
	trainY, testY := splitDataset(labels, 0.8)

	kNN := KNearestNeighbor{
		dataTrainX: trainX,
		dataTrainY: trainY,
		dataTestX:  testX,
		dataTestY:  testY,
	}
	score := kNN.score(3)
	fmt.Printf("Accuracy: %.2f\n", score)
}
