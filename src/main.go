package main

import (
    "fmt"
    "log"
    "os"
    "encoding/csv"
    "time"
    "kawAI/src/model"
	"strconv"
)

func main() {

	args := os.Args

	if len(args) < 2 {
		fmt.Println("No arguments provided.")
		return 
	}

	// args[2]
	file, err := os.Open("data/"+ args[1] +".csv")
	if err != nil {
		log.Fatalf("Failed to open CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	data, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Failed to read CSV file: %v", err)
	}

	startTime := time.Now()
	if (args[2] == "KNN") {

		intValue, err := strconv.Atoi(args[3])
		if err != nil {
			fmt.Println("Error converting string to int:", err)
			return
		}
		// fmt.Println(args[3])
		model.MainKNN(intValue, data[1:])
	}
	if (args[2] == "NB") {
		model.MainNB(data[1:])
	}
	
	if (args[2] == "ID3") {
		// fmt.Println("ID3")
		model.MainID3(data)
	}

	// model.MainID3(data)
	// model.MainNB(data[1:])
	fmt.Printf("Execution Time: %v\n", time.Since(startTime))
}