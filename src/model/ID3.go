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

func (id3 *ID3) indexOf(slice []string, element string) int {
	for i, v := range slice {
		if v == element {
			return i
		}
	}
	return -1
}

func (id3 *ID3) uniqueValues(dataX [][]string, dataY [][]string, att_index int, feature bool) (unique []string, counts map[string]int) {
	/*
		Returns map of unique values and their counts
		params:
		- att_index: index of an attribute to get the unique values 
		- data: if true then feature else target
	*/
	counts = make(map[string]int)
	var result []string
	if(feature){
		for _, value := range dataX {
			counts[value[att_index]]++
			if counts[value[att_index]] == 1 {
				result = append(result, value[att_index])
			}
		}
	} else {
		for _, value := range dataY {
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

func (id3 *ID3) entropyAttribute(dataX [][]string, dataY [][]string, att_index int) float64 {
	/*
		Returns entropy value of a attribute subset (subset is a value of the attribute)
		params:
		- att_index: index of an attribute to get the unique values 
	*/
	entropy := 0.0
	classes, _ := id3.uniqueValues(dataX, dataY, 0, false)
	attributes, counts := id3.uniqueValues(dataX, dataY, att_index, true)
	att_map := make(map[AttributeValue]int)
	for i := 0; i < len(dataX); i++ {
		att_map[AttributeValue{value: dataX[i][att_index], class: dataY[i][0]}]++
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

func (id3 *ID3) entropyClass(dataY [][]string) float64 {
	/*
		Returns E(S)
		params:
		- 
	*/
	entropy := 0.0
	_, counts := id3.uniqueValues([][]string{}, dataY, 0, false)
	for _, value := range counts {
		p := float64(float64(value) / float64(len(dataY)))
		entropy += -p * math.Log2(p)
	}

	return entropy
}

func (id3 *ID3) infoGain(dataX [][]string, dataY [][]string) string {
	/*
		Returns entropy value of a attribute subset (subset is a value of the attribute)
		params:
		- att_index: index of an attribute to get the unique values 
	*/
	map_gain := make(map[string]float64)
	entropy_class := id3.entropyClass(dataY)
	for i := 0; i < len(dataX[0]); i++ {
		map_gain[id3.columns[i]] = entropy_class - id3.entropyAttribute(dataX, dataY, i)
	}

	maxKey := ""
	var maxValue float64
	for key, value := range map_gain {
		if value > maxValue || maxKey == "" {
			maxValue = value
			maxKey = key
		}
	}

	return maxKey
}

type DTreeNode struct {
	attribute string
	decision map[string]*DTreeNode
	class string
}

func (id3 *ID3) filterData(dataX [][]string, dataY [][]string, attribute string, att_index int) ([][]string, [][]string) {
	filtered_data_x := [][]string{}
	filtered_data_y := [][]string{}
	for i := 0; i < len(dataX); i++ {
		if dataX[i][att_index] == attribute {
			filtered_data_x = append(filtered_data_x, dataX[i])
			filtered_data_y = append(filtered_data_y, dataY[i])
		}
	}
	
	return filtered_data_x, filtered_data_y 
}

func (id3 *ID3) recursiveBuildID3(node *DTreeNode, dataX [][]string, dataY [][]string) {
	if node.decision == nil {
        node.decision = make(map[string]*DTreeNode)
    }

	attribute := id3.infoGain(dataX, dataY)
	node.attribute = attribute
	att_index := id3.indexOf(id3.columns, attribute)
	attributes, _ := id3.uniqueValues(dataX, dataY, att_index, true)
	classes, _ := id3.uniqueValues(dataX, dataY, 0, false)
	if (len(classes) == 1) { // the node already has a class
		node.class = classes[0]
		// fmt.Println("result =>", node.class)
		return
	}

	for _, val := range attributes {
		// fmt.Println(attribute, "=> node")
		// fmt.Println(val)
		node.decision[val] = &DTreeNode{}
		filtered_x, filtered_y := id3.filterData(dataX, dataY, val, att_index)
		id3.recursiveBuildID3(node.decision[val], filtered_x, filtered_y)
	}
}

func (id3 *ID3) predict(input []string, dtree DTreeNode) string {
    node := &dtree
    att_index := id3.indexOf(id3.columns, dtree.attribute)

    for node.class == "" {
        // fmt.Println(node.attribute, "node")
		// fmt.Println(input[att_index], "val")
        nextNode := node.decision[input[att_index]]
        node = nextNode
		att_index = id3.indexOf(id3.columns, node.attribute)
    }

    return node.class
}


func (id3 *ID3) score() float64 {
	dtree := &DTreeNode{}
	id3.recursiveBuildID3(dtree, id3.dataTrainX, id3.dataTrainY)
	np := 0
	for i := 0; i < len(id3.dataTestX); i++ {
		predict := id3.predict(id3.dataTestX[i], *dtree)
		if(predict == id3.dataTestY[i][0]) {
			np++
		}
	}
	
	return float64(float64(np) / float64(len(id3.dataTestX)))
}

func MainID3(data [][]string) {
	var features [][]string
	var labels [][]string
	for _, row := range data[1:] {
		features = append(features, row[:len(row)-1])
		labels = append(labels, []string{row[len(row)-1]})
	}

	// trainX := features
	// testX := features
	// trainY := labels
	// testY := labels
	trainX, testX := splitDataset(features, 0.8)
	trainY, testY := splitDataset(labels, 0.8)

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