package main

import (
    "fmt"
    "log"
    "os"
    "encoding/csv"
    "time"
    "kawAI/src/model"
)

func main() {

	// args := os.Args

	// if len(args) < 2 {
	// 	fmt.Println("No arguments provided.")
	// 	return 
	// }

	// args[2]
	file, err := os.Open("data/"+ "data" +".csv")
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
	// if (args[3] == "KNN") {
	// 	model.MainKNN(data[1:])
	// }
	// if (args[3] == "NB") {
	// 	model.MainNB(data[1:])
	// }
	
	// if (args[3] == "ID3") {
	// 	// fmt.Println("ID3")
	// 	model.MainID3(data)
	// }

	model.MainID3(data)
	fmt.Printf("Execution Time: %v\n", time.Since(startTime))
}