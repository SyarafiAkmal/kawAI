package model

import (
	"fmt"
	"math"
	// "sort"
	// "sync"
)

type ID3 struct {
	dataTrainX [][]string
	dataTrainY [][]string
	dataTestX  [][]string
	dataTestY  [][]string
	columns []string
}

func (id3 *ID3) uniqueValues(att_index int, feature bool) (unique []string, counts map[string]int) {
	/*
		Returns map of unique values and their counts
		params:
		- att_index: index of an attribute to get the unique values 
		- data: if true then feature else target
	*/
	counts = make(map[string]int)
	var result []string
	if(feature){
		for _, value := range id3.dataTrainX {
			counts[value[att_index]]++
			if counts[value[att_index]] == 1 {
				result = append(result, value[att_index])
			}
		}
	} else {
		for _, value := range id3.dataTrainY {
			counts[value[att_index]]++
			if counts[value[att_index]] == 1 {
				result = append(result, value[att_index])
			}
		}
	}

	return result, counts
}

type AttributeValue struct {
	value string
	class string
}

func (id3 *ID3) entropyAttribute(att_index int) float64 {
	/*
		Returns entropy value of a attribute subset (subset is a value of the attribute)
		params:
		- att_index: index of an attribute to get the unique values 
	*/
	entropy := 0.0
	classes, _ := id3.uniqueValues(0, false)
	attributes, counts := id3.uniqueValues(att_index, true)
	// fmt.Println(attributes)
	att_map := make(map[AttributeValue]int)
	for i := 0; i < len(id3.dataTrainX); i++ {
		att_map[AttributeValue{value: id3.dataTrainX[i][att_index], class: id3.dataTrainY[i][0]}]++
	}
	
	for _, class := range classes {
		for _, attribute := range attributes {
			_, exists := att_map[AttributeValue{value: attribute, class: class}]
			if !exists {
				att_map[AttributeValue{value: attribute, class: class}] = 0
			}
		}
	}
	// fmt.Println(att_map)

	for _, attribute := range attributes {
		entropy_sub := 0.0
		for key, value := range att_map {
			if key.value == attribute {
				p := float64(float64(value) / float64(counts[attribute]))
				if p == 0 {
					entropy_sub += 0
				} else {
					entropy_sub += -p * math.Log2(p)
				}
			}
		}
		p_attribute := float64(float64(counts[attribute]) / float64(len(id3.dataTrainX)))
		entropy += p_attribute * entropy_sub
	}

	return entropy
}

func (id3 *ID3) entropyClass() float64 {
	/*
		Returns E(S)
		params:
		- 
	*/
	entropy := 0.0
	_, counts := id3.uniqueValues(0, false)
	for _, value := range counts {
		p := float64(float64(value) / float64(len(id3.dataTrainY)))
		entropy += -p * math.Log2(p)
	}

	return entropy
}

func (id3 *ID3) infoGain() float64 {
	map_gain := make(map[string]float64)
	entropy_class := id3.entropyClass()
	for i := 0; i < len(id3.dataTrainX[0]); i++ {
		map_gain[id3.columns[i]] = entropy_class - id3.entropyAttribute(i)
	}
	fmt.Println(map_gain)
	return 0.0
}

func (id3 *ID3) score() float64 {
	score := id3.infoGain()
	// fmt.Println(id3.columns)

	return score
}

func MainID3(data [][]string) {

	var features [][]string
	var labels [][]string
	for _, row := range data[1:] {
		features = append(features, row[:len(row)-1])
		labels = append(labels, []string{row[len(row)-1]})
	}

	trainX := features
	testX := features
	trainY := labels
	testY := labels
	// trainX, testX := splitDataset(features, 0.8)
	// trainY, testY := splitDataset(labels, 0.8)

	ID3 := ID3{
		dataTrainX: trainX,
		dataTrainY: trainY,
		dataTestX:  testX,
		dataTestY:  testY,
		columns: data[:1][0],
	}

	score := ID3.score()
	fmt.Printf("Accuracy: %.2f\n", score)
}