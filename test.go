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
    FeeBase     float64    // base fee in satoshis
    FeeRate     float64    // fee rate in ppm
    HopCost     float64
    Amt         float64
    CLTV        float64
    Capacity    float64
    HtlcMin     float64    // HTLC minimum in satoshis
    HtlcMax     float64    // HTLC maximum in satoshis
    SuccessProb float64
}

// Graph represents the Lightning Network as an adjacency list
type Graph struct {
	Nodes map[string][]Channel
}

// Result stores pathfinding results
type Result struct {
    Source          string
    Destination     string
    WeightFunction  string
    ProbabilityModel string
    Amount          float64
    PathLength      int
    TotalFee        float64
    TotalHopCost    float64
    Success         bool
    Path            string
}

// item represents an element in the priority queue
type Item struct {
	node     		string
	additiveCost	float64
	multiplicative  float64
	index    		int
	path     		[]Channel	
}

// priority queue implements heap.Interface
type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { 
	return (pq[i].additiveCost + pq[i].multiplicative) < (pq[j].additiveCost + pq[j].multiplicative)
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


// bimodal success probability should be integrated

// Estimate success probability using a bimodal function
// make s customizable
// area under curve must equal 1
// s = cap/10
func EstimateSuccessProbability(ch Channel) float64 {
	s := ch.Capacity / 10.0
    Pe := math.Exp(-ch.Amt/s) + math.Exp((ch.Amt-ch.Capacity)/s)
	return Pe
}

// estimate success probability using uniform function
// pe = (cap - amt) / cap
func EstimateUniformSuccessProbability(ch Channel) float64 {
	if ch.Capacity == 0 {
        return 0 // Avoid division by zero
    }
    return (ch.Capacity - ch.Amt) / ch.Capacity
}


type ProbabilityModel string

const (
    Bimodal ProbabilityModel = "bimodal"
    Uniform ProbabilityModel = "uniform"
)


// LND weight function
func LNDWeight(ch Channel, currentProb float64, model ProbabilityModel) (float64, float64) {
    var Pe float64
    if model == Bimodal {
        Pe = EstimateSuccessProbability(ch)
    } else {
        Pe = EstimateUniformSuccessProbability(ch)
    }
    
    additive := ch.Fee + (ch.Amt * ch.CLTV * 15e-9)
    penalty := 100.0 + ch.Amt*1000.0 // in msat
    penalty /= 1000 // convert to satoshis
    
    multiplicative := penalty / (math.Max(currentProb * Pe, 1e-9)) // Avoid division by zero

    return additive, multiplicative
}

func EclairWeight(ch Channel, model ProbabilityModel) float64 {
    var Pe float64
    if model == Bimodal {
        Pe = EstimateSuccessProbability(ch)
    } else {
        Pe = EstimateUniformSuccessProbability(ch)
    }
    
    lockedFundsRisk := 1e-8 // per paper
    riskCost := ch.Amt * ch.CLTV * lockedFundsRisk
    
    return (ch.Fee + ch.HopCost + riskCost) - ((2000+ch.Amt*500)/1000)*math.Log(Pe) // /1000 to convert from msat to satoshi?
}

// CLN weight function as per the paper
// Note: CLN does NOT use success probability like other implementations
func CLNWeight(ch Channel) float64 {
    blockPerYear := 52596.0
    riskFactor := 10.0
	// this capacityBias is success prob in CLN
	estimateSuccessProb := (ch.Capacity + 1 - ch.Amt) / (ch.Capacity + 1)
    capacityBias := -math.Log(estimateSuccessProb)
    
    return (ch.Fee + (ch.Amt * ch.CLTV * riskFactor) / (blockPerYear * 100) + 1) * (capacityBias + 1)
}

// LDK weight function as per the paper
func LDKWeight(ch Channel, model ProbabilityModel) float64 {
    // Calculate pathHtlcMin
    pathHtlcMin := ch.HtlcMin*(1+ch.FeeRate/1000000) + ch.FeeBase
    
    // Calculate base penalty
    penaltyBase := 500.0 // msat
    baseMultiplier := 8192.0 // msat
    basePenalty := penaltyBase + (baseMultiplier * ch.Amt) / math.Pow(2, 30)
    
    // Calculate anti-probing penalty
    antiProbingPenalty := 0.0
    if ch.HtlcMax >= ch.Capacity/2 {
        antiProbingPenalty = 250.0 // msat
    }
    
    // Calculate liquidity penalty
    LM := 30000.0 // msat
    amtMultiplier := (192.0 * ch.Amt) / math.Pow(2, 20)
    var liquidityPenalty float64
    
    // Using uniform distribution for success probability
	// not using exponential in bimodal; instead is polynomial
    var Pe float64
    if model == Bimodal {
        Pe = EstimateSuccessProbability(ch)
    } else {
        Pe = EstimateUniformSuccessProbability(ch)
    }
    liquidityPenalty = -math.Log10(Pe) * (LM + amtMultiplier)
    
    // Calculate historic penalty (simplified - paper mentions historical data)
    HM := 10000.0 // msat
    historicMultiplier := (64.0 * ch.Amt) / math.Pow(2, 20)
    historicPenalty := -math.Log10(Pe) * (HM + historicMultiplier)
    
    // Total penalty
    penalty := basePenalty + antiProbingPenalty + liquidityPenalty + historicPenalty
    
    // Convert penalties from msat to satoshis
    penalty /= 1000
    
    return math.Max(ch.Fee, pathHtlcMin) + penalty
}

func CombinedWeight(ch Channel, currentProb float64, model ProbabilityModel) (float64, float64) {
    // implement with bimodal exponential vs polynomial
	var Pe float64
    if model == Bimodal {
        Pe = EstimateSuccessProbability(ch)
    } else {
        Pe = EstimateUniformSuccessProbability(ch)
    }
    
    // Eclair-inspired components
    riskCost := ch.Amt * ch.CLTV * 1.8e-8
    additive := (ch.Fee + ch.HopCost + riskCost) * 1.2
    
    // LND-inspired penalty term
    penalty := (2000 + ch.Amt*500)
    multiplicative := penalty / (currentProb * Pe)
    
    return additive, multiplicative
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
        from := strings.TrimSpace(record[1]) // source node
        to := strings.TrimSpace(record[2])   // destination node

        // convert fields from string to appropriate data types
        capacity, _ := strconv.ParseFloat(record[5], 64) // satoshis
        
        // Convert amount from msat to satoshis
        amountStr := strings.TrimSuffix(record[6], "msat")
        amt, _ := strconv.ParseFloat(amountStr, 64)
        amt /= 1000 // convert msat to satoshis

        // Fee components
        feeBase, _ := strconv.ParseFloat(record[11], 64) // base fee in msat
        feeBase /= 1000                                  // convert msat to satoshis
        feeRate, _ := strconv.ParseFloat(record[12], 64) // ppm (per million)
        
        // CLTV delay
        delay, _ := strconv.ParseFloat(record[13], 64)   // delay (CLTV)

        // HTLC minimum/maximum (convert from msat to satoshis)
        htlcMinStr := strings.TrimSuffix(record[14], "msat")
        htlcMin, _ := strconv.ParseFloat(htlcMinStr, 64)
        htlcMin /= 1000
        
        htlcMaxStr := strings.TrimSuffix(record[15], "msat")
        htlcMax, _ := strconv.ParseFloat(htlcMaxStr, 64)
        htlcMax /= 1000

        // calculate total fee using fee formula: base_fee + (amt * fee_rate / 1,000,000)
        fee := feeBase + (amt * feeRate / 1000000)

        // construct channel struct with all LDK required fields
        channel := Channel{
            FromNode:    from,
            ToNode:      to,
            Fee:         fee,
            FeeBase:     feeBase,    // Added for LDK
            FeeRate:     feeRate,    // Added for LDK
            HopCost:     delay,
            Amt:         amt,
            CLTV:        delay,
            Capacity:    capacity,
            HtlcMin:     htlcMin,    // Added for LDK
            HtlcMax:     htlcMax,    // Added for LDK
            SuccessProb: 0.95,       // default success probability
        }

        // add channel to the graph
        graph.Nodes[from] = append(graph.Nodes[from], channel)
    }

    return graph, nil
}


// Modified Dijkstra's algorithm
func FindBestRoute(graph *Graph, start, end string, amount float64, weightFunc func(Channel, float64) (float64, float64)) ([]Channel, error) {
    pq := &PriorityQueue{}
    heap.Init(pq)

    // Track both additive and multiplicative costs
    distAdd := make(map[string]float64)
    distMult := make(map[string]float64)
    prev := make(map[string]Channel)
    path := make(map[string][]Channel)
    visited := make(map[string]bool)

    // Initialize distances
    for node := range graph.Nodes {
        distAdd[node] = math.Inf(1)
        distMult[node] = 1.0
    }
    distAdd[start] = 0
    distMult[start] = 1.0
    path[start] = []Channel{}

    heap.Push(pq, &Item{
        node:           start,
        additiveCost:   0,
        multiplicative: 0,
        path:           []Channel{},
    })

    maxIterations := len(graph.Nodes) * 100 // Prevent infinite loops
    iterations := 0

    for pq.Len() > 0 && iterations < maxIterations {
        iterations++
        current := heap.Pop(pq).(*Item)
        
        if current.node == end {
            return path[current.node], nil
        }

        if visited[current.node] {
            continue
        }
        visited[current.node] = true

        for _, ch := range graph.Nodes[current.node] {
            // Skip channels that can't handle the amount
            if ch.Amt < amount*0.9 || ch.Capacity < amount {
                continue
            }

            // Calculate current cumulative success probability along path
            currentProb := distMult[current.node]
            if currentProb <= 0 {
                currentProb = 1e-9 // Avoid division by zero
            }

            // Get weights from the weight function
            additive, multiplicative := weightFunc(ch, currentProb)
            
            // Validate weights
            if math.IsNaN(additive) || math.IsInf(additive, 0) ||
                math.IsNaN(multiplicative) || math.IsInf(multiplicative, 0) {
                continue
            }

            newAdd := distAdd[current.node] + additive
            newMult := distMult[current.node] * math.Max(multiplicative, 1e-9)
            
            // Compare total cost
            if (newAdd + newMult) < (distAdd[ch.ToNode] + distMult[ch.ToNode]) {
                distAdd[ch.ToNode] = newAdd
                distMult[ch.ToNode] = newMult
                prev[ch.ToNode] = ch
                
                // Update path
                newPath := make([]Channel, len(path[current.node]))
                copy(newPath, path[current.node])
                newPath = append(newPath, ch)
                path[ch.ToNode] = newPath
                
                heap.Push(pq, &Item{
                    node:           ch.ToNode,
                    additiveCost:   newAdd,
                    multiplicative: newMult,
                    path:           newPath,
                })
            }
        }
    }

    return nil, fmt.Errorf("no path found after %d iterations", iterations)
}


// Helper function to wrap weight functions for Dijkstra
func makeWeightFuncWrapper(weightFunc func(Channel, float64, ProbabilityModel) (float64, float64)) func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, error) {
        return FindBestRoute(graph, start, end, amount, func(ch Channel, prob float64) (float64, float64) {
            return weightFunc(ch, prob, model)
        })
    }
}

func makeEclairWrapper(weightFunc func(Channel, ProbabilityModel) float64) func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, error) {
        return FindBestRoute(graph, start, end, amount, func(ch Channel, _ float64) (float64, float64) {
            return weightFunc(ch, model), 0
        })
    }
}

func makeLDKWrapper(weightFunc func(Channel, ProbabilityModel) float64) func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, error) {
        return FindBestRoute(graph, start, end, amount, func(ch Channel, _ float64) (float64, float64) {
            return weightFunc(ch, model), 0
        })
    }
}

func makeCLNWrapper() func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, error) {
    return func(g *Graph, s, e string, a float64, _ ProbabilityModel) ([]Channel, error) {
        return FindBestRoute(g, s, e, a, func(ch Channel, _ float64) (float64, float64) {
            return CLNWeight(ch), 0
        })
    }
}


func SaveComprehensiveResults(results []Result, filename string) {
    file, err := os.Create(filename)
    if err != nil {
        fmt.Println("Error creating CSV file:", err)
        return
    }
    defer file.Close()

    writer := csv.NewWriter(file)
    defer writer.Flush()

    // Write header
    writer.Write([]string{
        "Source", "Destination", "WeightFunction", "ProbabilityModel", 
        "Amount", "Success", "PathLength", "TotalFee", 
        "TotalHopCost", "Path",
    })

    // Write data
    for _, result := range results {
        writer.Write([]string{
            result.Source,
            result.Destination,
            result.WeightFunction,
            result.ProbabilityModel,
            fmt.Sprintf("%.6f", result.Amount),
            strconv.FormatBool(result.Success),
            strconv.Itoa(result.PathLength),
            fmt.Sprintf("%.6f", result.TotalFee),
            fmt.Sprintf("%.6f", result.TotalHopCost),
            result.Path,
        })
    }
    fmt.Println("Comprehensive results saved to", filename)
}


func ComprehensiveTest(graph *Graph, numTests int) {
    rand.Seed(time.Now().UnixNano())
    results := []Result{}
    
    for i := 0; i < numTests; i++ {
        src, dst := RandomNodes(graph)
        if src == dst || src == "" || dst == "" {
            continue
        }
        amount := rand.Float64()*100000 + 1000
        
        for _, model := range []ProbabilityModel{Bimodal, Uniform} {
            for _, testCase := range []struct {
				Name string
				Func func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, error)
				UsesProbabilityModel bool
			}{
				{"LND", makeWeightFuncWrapper(LNDWeight), true},
				{"Eclair", makeEclairWrapper(EclairWeight), true},
				{"CLN", makeCLNWrapper(), false},
				{"LDK", makeLDKWrapper(LDKWeight), true},
				{"Combined", makeWeightFuncWrapper(CombinedWeight), true},
			} {
                // For CLN, we pass empty model since it's not used
                var modelArg ProbabilityModel
                if testCase.UsesProbabilityModel {
                    modelArg = model
                }
                
                path, err := testCase.Func(graph, src, dst, amount, modelArg)
                
                // Create result entry
                result := Result{
                    Source:          src,
                    Destination:     dst,
                    WeightFunction:  testCase.Name,
                    ProbabilityModel: string(modelArg), // Will be empty for CLN
                    Amount:          amount,
                    Success:         err == nil,
                }
                
                if !testCase.UsesProbabilityModel {
                    result.ProbabilityModel = "N/A (capacity bias)"
                }
                
                if err == nil {
                    totalFee, totalHopCost := 0.0, 0.0
                    pathStr := ""
                    for _, ch := range path {
                        totalFee += ch.Fee
                        totalHopCost += ch.HopCost
                        pathStr += fmt.Sprintf("%s -> ", ch.FromNode)
                    }
                    pathStr += dst
                    
                    result.PathLength = len(path)
                    result.TotalFee = totalFee
                    result.TotalHopCost = totalHopCost
                    result.Path = pathStr
                }
                
                results = append(results, result)
            }
        }
    }
    
    SaveComprehensiveResults(results, "comprehensive_results.csv")
}


func main() {
    fmt.Println("Loading network data...")
    graph, err := LoadCSV("LN_snapshot.csv")
    if err != nil {
        fmt.Println("Error loading network data:", err)
        return
    }

    fmt.Printf("Loaded network with %d nodes\n", len(graph.Nodes))
    fmt.Println("Running tests...")

    numTests := 1 // Start with 1 test for debugging
    ComprehensiveTest(graph, numTests)
    fmt.Println("Tests completed")
}
