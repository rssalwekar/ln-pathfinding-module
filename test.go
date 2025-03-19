package main

import (
	"container/heap"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"math/rand"
	"time"
)

// Channel represents a Lightning Network channel
type Channel struct {
	FromNode    string
	ToNode      string
	Fee         float64
	HopCost     float64
	Amt         float64
	CLTV        float64
	Capacity    float64
	SuccessProb float64
}

// Graph represents the Lightning Network as an adjacency list
type Graph struct {
	Nodes map[string][]Channel
}

// Result stores pathfinding results, now includes transaction amount
type Result struct {
	Source         string
	Destination    string
	WeightFunction string
	Amount         float64 // New field to store transaction amount
	PathLength     int
	TotalFee       float64
	TotalHopCost   float64
	Path           string
}

// item represents an element in the priority queue
type Item struct {
	node     string
	priority float64
	index    int
}

// priority queue implements heap.Interface
type PriorityQueue []*Item

func (pq PriorityQueue) Len() int           { 
	return len(pq) 
}
func (pq PriorityQueue) Less(i, j int) bool { 
	return pq[i].priority < pq[j].priority 
}
func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index, pq[j].index = i, j
}

func (pq *PriorityQueue) Push(x interface{}) {
	item := x.(*Item)
	item.index = len(*pq)
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[:n-1]
	return item
}

// Selects two random **connected** nodes
func RandomNodes(graph *Graph) (string, string) {
	nodes := make([]string, 0, len(graph.Nodes))
	for node := range graph.Nodes {
		if len(graph.Nodes[node]) > 0 {
			nodes = append(nodes, node)
		}
	}
	if len(nodes) < 2 {
		return "", ""
	}

	var src, dst string
	for i := 0; i < 100; i++ {
		src = nodes[rand.Intn(len(nodes))]
		dst = nodes[rand.Intn(len(nodes))]
		if src != dst {
			return src, dst
		}
	}
	return "", ""
}

// Estimate success probability using a bimodal function
func EstimateSuccessProbability(ch Channel) float64 {
	s := 300000.0
	Pe := math.Exp(-ch.Amt/s) * math.Exp((ch.Amt-ch.Capacity)/s)
	return math.Max(0.01, Pe)
}

func LNDWeight(ch Channel) float64 {
	ch.SuccessProb = EstimateSuccessProbability(ch)
	penalty := (2000 + ch.Amt*500) * 2
	return ch.Fee*3.0 + (ch.Amt * ch.CLTV * 20e-9) + (penalty / ch.SuccessProb)
}

func EclairWeight(ch Channel) float64 {
	ch.SuccessProb = EstimateSuccessProbability(ch)
	riskCost := ch.Amt * ch.CLTV * 2.0e-8
	return (ch.Fee + ch.HopCost + riskCost) + ((2000 + ch.Amt*500) / ch.SuccessProb)
}

func CombinedWeight(ch Channel) float64 {
	ch.SuccessProb = EstimateSuccessProbability(ch)
	riskCost := ch.Amt * ch.CLTV * 1.8e-8
	return (ch.Fee + ch.HopCost + riskCost) * 1.2 + ((2000 + ch.Amt*500) / ch.SuccessProb)
}

// loads the lightning network data from the CSV file
func LoadCSV(filename string) (*Graph, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	graph := &Graph{Nodes: make(map[string][]Channel)}

	for _, record := range records[1:] { // skip header row
		from := record[1] // source node
		to := record[2]   // destination node

		// convert fields from string to appropriate data types
		capacity, _ := strconv.ParseFloat(record[5], 64) // satoshis
		amountStr := strings.TrimSuffix(record[6], "msat")
		amt, _ := strconv.ParseFloat(amountStr, 64)
		amt /= 1000 // convert msat to satoshis

		feeBase, _ := strconv.ParseFloat(record[11], 64) // base fee in msat
		feeBase /= 1000                                  // convert msat to satoshis

		feeRate, _ := strconv.ParseFloat(record[12], 64) // ppm (per million)
		delay, _ := strconv.ParseFloat(record[13], 64)   // delay (CLTV)

		// calculate total fee using fee formula: base_fee + (amt * fee_rate / 1,000,000)
		fee := feeBase + (amt * feeRate / 1000000)

		// construct channel struct
		channel := Channel{
			FromNode:    from,
			ToNode:      to,
			Fee:         fee,
			HopCost:     delay,
			Amt:         amt,
			CLTV:        delay,
			Capacity:    capacity,
			SuccessProb: 0.95, // assume a default success probability
		}

		// add channel to the graph
		graph.Nodes[from] = append(graph.Nodes[from], channel)
	}

	return graph, nil
}

// STEP 2: pathfinding
// implement dijkstra's algorithm with a dynamic weight function
func FindBestRoute(graph *Graph, start string, end string, amount float64, weightFunc func(Channel) float64) ([]Channel, error) {
	pq := &PriorityQueue{}
	heap.Init(pq)

	dist := make(map[string]float64)
	prev := make(map[string]Channel)

	for node := range graph.Nodes {
		dist[node] = math.Inf(1)
	}
	dist[start] = 0

	heap.Push(pq, &Item{node: start, priority: 0})

	for pq.Len() > 0 {
		current := heap.Pop(pq).(*Item)

		if current.node == end {
			break
		}

		for _, ch := range graph.Nodes[current.node] {
			if ch.Amt < amount * 0.9 { // ensure capacity is enough with a 10% buffer
				continue
			}

			weight := weightFunc(ch)

			newDist := dist[current.node] + weight
			if newDist < dist[ch.ToNode] {
				dist[ch.ToNode] = newDist
				prev[ch.ToNode] = ch
				heap.Push(pq, &Item{node: ch.ToNode, priority: newDist})
			}
		}
	}

	// reconstruct path
	path := []Channel{}
	for at := end; at != start; {
		ch, ok := prev[at]
		if !ok {
			return nil, fmt.Errorf("no path found")
		}
		path = append([]Channel{ch}, path...)
		at = ch.FromNode
	}

	return path, nil
}

// Save results to a CSV file, now including transaction amount
func SaveResultsToCSV(results []Result, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		fmt.Println("Error creating CSV file:", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"Source", "Destination", "Weight Function", "Transaction Amount", "Path Length", "Total Fee", "Total Hop Cost", "Path"})
	for _, result := range results {
		writer.Write([]string{
			result.Source, result.Destination, result.WeightFunction,
			fmt.Sprintf("%.6f", result.Amount),
			strconv.Itoa(result.PathLength), fmt.Sprintf("%.6f", result.TotalFee),
			fmt.Sprintf("%.6f", result.TotalHopCost), result.Path,
		})
	}
	fmt.Println("Results saved to", filename)
}

func BatchPathfinding(graph *Graph, numTests int) {
	rand.Seed(time.Now().UnixNano())
	results := []Result{}

	for i := 0; i < numTests; i++ {
		src, dst := RandomNodes(graph)
		if src == dst || src == "" || dst == "" {
			continue
		}
		amount := rand.Float64()*10000 + 1000 // Transaction amount range: 1000 - 6000

		for _, weightFunc := range []struct {
			Name string
			Func func(Channel) float64
		}{
			{"LND", LNDWeight},
			{"Eclair", EclairWeight},
			{"Combined", CombinedWeight},
		} {
			path, err := FindBestRoute(graph, src, dst, amount, weightFunc.Func)
			if err != nil {
				fmt.Println("Error finding path:", err)
				continue
			}

			totalFee, totalHopCost := 0.0, 0.0
			pathStr := ""
			for _, ch := range path {
				totalFee += ch.Fee
				totalHopCost += ch.HopCost
				pathStr += fmt.Sprintf("%s -> ", ch.FromNode)
			}
			pathStr += dst

			results = append(results, Result{
				Source: src, Destination: dst, WeightFunction: weightFunc.Name,
				Amount: amount, PathLength: len(path), TotalFee: totalFee,
				TotalHopCost: totalHopCost, Path: pathStr,
			})
		}
	}
	SaveResultsToCSV(results, "pathfinding_results.csv")
}

func main() {
	graph, err := LoadCSV("LN_snapshot.csv")
	if err != nil {
		fmt.Println("Error loading network data:", err)
		return
	}

	numTests := 20
	BatchPathfinding(graph, numTests)
}
