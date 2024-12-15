package model

import (
	"fmt"
	"log"
	"math"
	"sync"
)

type NaiveBayes struct {
	dataTrainX [][]string
	dataTrainY [][]string
	dataTestX  [][]string
	dataTestY  [][]string
	model      map[string]map[string]map[string]float64
}

// Function to train the Naive Bayes model
func (nb *NaiveBayes) fit() {
	// Collect unique attribute values
	attributeVals := make(map[string]bool)
	for _, row := range append(nb.dataTrainX, nb.dataTestX...) {
		for _, cell := range row {
			attributeVals[cell] = true
		}
	}

	// Get unique classes
	classCounts := make(map[string]int)
	for _, row := range nb.dataTrainY {
		classCounts[row[0]]++
	}

	// fmt.Println("Fitting...")
	// Initialize the model
	nb.model = make(map[string]map[string]map[string]float64)
	for class, freq := range classCounts {
		nb.model[class] = make(map[string]map[string]float64)
		nb.model[class]["class"] = map[string]float64{
			"freq": float64(freq),
			"p":    float64(freq) / float64(len(nb.dataTrainY)),
		}
		for attr := range attributeVals {
			nb.model[class][attr] = map[string]float64{
				"freq": 0,
				"p":    0,
			}
		}
	}

	// Populate frequencies
	for i, row := range nb.dataTrainX {
		class := nb.dataTrainY[i][0]
		for _, cell := range row {
			if _, exists := nb.model[class][cell]; exists {
				nb.model[class][cell]["freq"]++
			}
		}
	}

	// Calculate probabilities
	for class := range nb.model {
		for attr := range nb.model[class] {
			if attr != "class" {
				nb.model[class][attr]["p"] = nb.model[class][attr]["freq"] / nb.model[class]["class"]["freq"]
			}
		}
	}

	// for key, innerMap := range nb.model {
	// 	fmt.Printf("%s:\n", key)
	// 	for innerKey, subMap := range innerMap {
	// 		fmt.Printf("\t%s:\n", innerKey)
	// 		for subKey, value := range subMap {
	// 			fmt.Printf("\t\t%s: %.2f\n", subKey, value)
	// 		}
	// 	}
	// }
}


func (nb *NaiveBayes) predict(query []string) string {
	classes := make([]string, 0)
	for class := range nb.model {
		classes = append(classes, class)
	}

	if len(classes) == 0 {
		log.Fatal("Model is empty. Ensure fit() is called and training data is valid.")
	}

	results := make([][2]float64, len(classes))
	for i, class := range classes {
		p := nb.model[class]["class"]["p"]
		for _, cell := range query {
			if prob, exists := nb.model[class][cell]["p"]; exists {
				p *= prob
			}
		}
		results[i] = [2]float64{float64(i), p}
	}

	maxIndex := 0
	maxValue := results[0][1]
	for i, result := range results {
		if result[1] > maxValue {
			maxIndex = i
			maxValue = result[1]
		}
	}

	return classes[maxIndex]
}

func (nb *NaiveBayes) score() float64 {
	var wg sync.WaitGroup
	var mu sync.Mutex

	nb.fit()
	tp := 0
	
	for i, row := range nb.dataTestX {
		wg.Add(1)
		go func (row []string, i int)  {
			wg.Done()
			if nb.predict(row) == nb.dataTestY[i][0] {
				mu.Lock()
				tp++
				mu.Unlock()
			}
		}(row, i)
	}
	wg.Wait()
	
	fmt.Println(tp, "/",len(nb.dataTestX), "correct")
	return float64(tp) / float64(len(nb.dataTestX))
}

func splitDataset(data [][]string, splitRatio float64) ([][]string, [][]string) {
	splitIndex := int(math.Floor(float64(len(data)) * splitRatio))
	return data[:splitIndex], data[splitIndex:]
}

func MainNB(data [][]string) {

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
	trainX, testX := splitDataset(features, 0.5)
	trainY, testY := splitDataset(labels, 0.5)

	nb := NaiveBayes{
		dataTrainX: trainX,
		dataTrainY: trainY,
		dataTestX:  testX,
		dataTestY:  testY,
	}	
	score := nb.score()
	fmt.Printf("Accuracy: %.2f\n", score)
}
