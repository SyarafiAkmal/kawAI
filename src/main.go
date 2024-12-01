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
	file, err := os.Open("data/data.csv")
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
	model.MainNB(data[1:])
	fmt.Printf("Execution Time: %v\n", time.Since(startTime))
}