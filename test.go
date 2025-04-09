package main

import (
	"container/heap"
	"encoding/csv"
	"flag"
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

// User preference flag: 0.0 = minimize cost, 1.0 = maximize reliability
var ReliabilityBias = flag.Float64("reliability", 0.1, "Preference for reliability: 0=low fee, 1=high reliability")

// Result struct includes success, path, fee, reliability, etc.
type Result struct {
	Source, Destination, WeightFunction, ProbabilityModel string
	Amount, TotalFee, TotalHopCost, PathPe                float64
	PathLength                                            int
	Path                                                  string
	Success                                               bool
	Retries                                               int
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

// Selects two random connected nodes
func RandomNodes(r *rand.Rand, graph *Graph) (string, string) {
	nodes := make([]string, 0, len(graph.Nodes))
	for node := range graph.Nodes {
		if len(graph.Nodes[node]) > 0 {
			nodes = append(nodes, node)
		}
	}
	if len(nodes) < 2 {
		return "", ""
	}

	for i := 0; i < 100; i++ {
		src := nodes[r.Intn(len(nodes))]
		dst := nodes[r.Intn(len(nodes))]
		if src != dst {
			return src, dst
		}
	}
	return "", ""
}

// Compute the max amount that can be sent from a node
func maxSendable(graph *Graph, node string) float64 {
	maxAmt := 0.0
	if chans, ok := graph.Nodes[node]; ok {
		for _, ch := range chans {
			if ch.Capacity > maxAmt {
				maxAmt = ch.Capacity
			}
		}
	}
	return maxAmt
}

// Compute the max amount that can be received by a node
func maxReceivable(graph *Graph, node string) float64 {
	maxAmt := 0.0
	for _, chans := range graph.Nodes {
		for _, ch := range chans {
			if ch.ToNode == node && ch.Capacity > maxAmt {
				maxAmt = ch.Capacity
			}
		}
	}
	return maxAmt
}


// bimodal success probability should be integrated
// Estimate success probability using a bimodal function
// make s customizable
// area under curve must equal 1
// s = cap/10
func EstimateBimodalSuccessProbability(ch Channel) float64 {
	if ch.Amt >= ch.Capacity || ch.Capacity == 0 {
		return 0.01
	}

	// Scale parameter s controls how "bimodal" the distribution is
	s := ch.Capacity / 10.0

	// PDF of bimodal distribution: exp(-x/s) + exp((x-cap)/s)
	pdf := func(x float64) float64 {
		return math.Exp(-x/s) + math.Exp((x - ch.Capacity)/s)
	}

	// Trapezoidal numerical integration
	integrate := func(f func(float64) float64, a, b float64, steps int) float64 {
		if b <= a {
			return 0
		}
		h := (b - a) / float64(steps)
		sum := 0.5 * (f(a) + f(b))
		for i := 1; i < steps; i++ {
			sum += f(a + float64(i)*h)
		}
		return sum * h
	}

	// Calculate success probability Pe = ∫_amt^cap pdf(x) dx / ∫_0^cap pdf(x) dx
	numerator := integrate(pdf, ch.Amt, ch.Capacity, 100)
	denominator := integrate(pdf, 0, ch.Capacity, 100)

	if denominator == 0 {
		return 0.01
	}

	Pe := numerator / denominator
	return math.Max(Pe, 0.000) // Cap minimum at 0.01%
}


// estimate success probability using uniform function
// pe = (cap - amt) / cap
func EstimateUniformSuccessProbability(ch Channel) float64 {
	if ch.Capacity == 0 {
        return 0 // Avoid division by zero
    }
    return math.Max((ch.Capacity - ch.Amt) / ch.Capacity, 0.000)
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
        Pe = EstimateBimodalSuccessProbability(ch)
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
        Pe = EstimateBimodalSuccessProbability(ch)
    } else {
        Pe = EstimateUniformSuccessProbability(ch)
    }
    
    lockedFundsRisk := 1e-8 // per paper
    riskCost := ch.Amt * ch.CLTV * lockedFundsRisk

	logPe := math.Log(Pe)
	penalty := ((2000 + ch.Amt*500) / 1000) * logPe
	total := (ch.Fee + ch.HopCost + riskCost) - penalty
    
    // return (ch.Fee + ch.HopCost + riskCost) - ((2000+ch.Amt*500)/1000)*math.Log(Pe) // /1000 to convert from msat to satoshi?
	return total
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
        Pe = EstimateBimodalSuccessProbability(ch)
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

func CombinedWeight(ch Channel, _ float64, model ProbabilityModel) (float64, float64) {
	var Pe float64
	if model == Bimodal {
		Pe = EstimateBimodalSuccessProbability(ch)
	} else {
		Pe = EstimateUniformSuccessProbability(ch)
	}
	Pe = math.Max(Pe, 0.01)
	riskCost := ch.Amt * ch.CLTV * 1.8e-8
	feeComponent := ch.Fee + ch.HopCost + riskCost
	combined := (1.0 - *ReliabilityBias)*feeComponent + *ReliabilityBias*(1.0/Pe)
	return combined, Pe
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

    distAdd := make(map[string]float64)        // additive cost F(p)
    distLogProb := make(map[string]float64)    // log success probability: sum log(Pe)
    prev := make(map[string]Channel)
    path := make(map[string][]Channel)
    visited := make(map[string]bool)

    // Initialize distances
    for node := range graph.Nodes {
        distAdd[node] = math.Inf(1)
        distLogProb[node] = 0.0 // log(1)
    }
    distAdd[start] = 0
    distLogProb[start] = 0.0
    path[start] = []Channel{}

    heap.Push(pq, &Item{
        node:           start,
        additiveCost:   0,
        multiplicative: 0, // use for logProb here (repurposed field)
        path:           []Channel{},
    })

    maxIterations := len(graph.Nodes) * 100
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
            if ch.Capacity < amount {
                continue
            }

            // Get weights (additive and success probability as "Pe")
            additive, multiplicative := weightFunc(ch, math.Exp(distLogProb[current.node]))

            // Interpret "multiplicative" as Pe for path probability
            Pe := math.Max(multiplicative, 1e-9) // avoid log(0)
            logPe := math.Log(Pe)

            if math.IsNaN(additive) || math.IsInf(additive, 0) || math.IsNaN(logPe) || math.IsInf(logPe, 0) {
                continue
            }

            newAdd := distAdd[current.node] + additive
            newLogProb := distLogProb[current.node] + logPe

            // Cost function: c(p) = additive + cattempt / exp(logP)
            cattempt := 1.0 // constant (per paper) — you can scale it if you want
            totalCost := newAdd + cattempt/math.Exp(newLogProb)

            // Compare total cost with existing path
            oldTotalCost := distAdd[ch.ToNode] + cattempt/math.Exp(distLogProb[ch.ToNode])
            if totalCost < oldTotalCost {
                distAdd[ch.ToNode] = newAdd
                distLogProb[ch.ToNode] = newLogProb
                prev[ch.ToNode] = ch

                newPath := make([]Channel, len(path[current.node]))
                copy(newPath, path[current.node])
                newPath = append(newPath, ch)
                path[ch.ToNode] = newPath

                heap.Push(pq, &Item{
                    node:           ch.ToNode,
                    additiveCost:   newAdd,
                    multiplicative: newLogProb, // repurposing field
                    path:           newPath,
                })
            }
        }
    }

    return nil, fmt.Errorf("no path found after %d iterations", iterations)
}


// Helper function to wrap weight functions for Dijkstra
func makeWeightFuncWrapper(weightFunc func(Channel, float64, ProbabilityModel) (float64, float64)) func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, float64, error) {
        path, err := FindBestRoute(graph, start, end, amount, func(ch Channel, prob float64) (float64, float64) {
            return weightFunc(ch, prob, model)
        })
        if err != nil {
            return nil, 0, err
        }
        pathPe := 1.0
        for _, ch := range path {
            var pe float64
            if model == Bimodal {
                pe = EstimateBimodalSuccessProbability(ch)
            } else {
                pe = EstimateUniformSuccessProbability(ch)
            }
            pathPe *= math.Max(pe, 1e-9)
        }
        return path, pathPe, nil
    }
}

func makeEclairWrapper(weightFunc func(Channel, ProbabilityModel) float64) func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, float64, error) {
        path, err := FindBestRoute(graph, start, end, amount, func(ch Channel, _ float64) (float64, float64) {
            return weightFunc(ch, model), 0
        })
        if err != nil {
            return nil, 0, err
        }
        pathPe := 1.0
        for _, ch := range path {
            var pe float64
            if model == Bimodal {
                pe = EstimateBimodalSuccessProbability(ch)
            } else {
                pe = EstimateUniformSuccessProbability(ch)
            }
            pathPe *= math.Max(pe, 1e-9)
        }
        return path, pathPe, nil
    }
}

func makeLDKWrapper(weightFunc func(Channel, ProbabilityModel) float64) func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, float64, error) {
        path, err := FindBestRoute(graph, start, end, amount, func(ch Channel, _ float64) (float64, float64) {
            return weightFunc(ch, model), 0
        })
        if err != nil {
            return nil, 0, err
        }
        pathPe := 1.0
        for _, ch := range path {
            var pe float64
            if model == Bimodal {
                pe = EstimateBimodalSuccessProbability(ch)
            } else {
                pe = EstimateUniformSuccessProbability(ch)
            }
            pathPe *= math.Max(pe, 1e-9)
        }
        return path, pathPe, nil
    }
}

func makeCLNWrapper() func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error) {
    return func(graph *Graph, start, end string, amount float64, _ ProbabilityModel) ([]Channel, float64, error) {
        path, err := FindBestRoute(graph, start, end, amount, func(ch Channel, _ float64) (float64, float64) {
            return CLNWeight(ch), 0
        })
        if err != nil {
            return nil, 0, err
        }
        return path, 1.0, nil // CLN doesn't use Pe
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
		"TotalHopCost", "Path", "Retries", "PathPe",
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
			strconv.Itoa(result.Retries),
			fmt.Sprintf("%.6f", result.PathPe),
		})
	}

    fmt.Println("Comprehensive results saved to", filename)
}


func ComprehensiveTest(graph *Graph, numTests int) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	results := []Result{}
	maxRetries := 1

	for i := 0; i < numTests; i++ {
		src, dst := RandomNodes(r, graph)
		if src == dst || src == "" || dst == "" {
			continue
		}
		
		// Limit by actual feasible amounts from snapshot graph
		maxSrcAmt := maxSendable(graph, src) // e.g., max of outgoing channel balances
		maxDstAmt := maxReceivable(graph, dst) // e.g., max of incoming channel balances
		feasibleAmt := math.Min(maxSrcAmt, maxDstAmt)
		if feasibleAmt < 1000 {
			continue // skip this test if too small to be meaningful
		}
		amount := r.Float64() * feasibleAmt  // push past edges
		
		for _, model := range []ProbabilityModel{Bimodal, Uniform} {
			for _, testCase := range []struct {
				Name                 string
				Func                 func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error)
				UsesProbabilityModel bool
			}{
				{"Combined", makeWeightFuncWrapper(CombinedWeight), true},
				{"Eclair", makeEclairWrapper(EclairWeight), true},
				{"CLN", makeCLNWrapper(), false},
				{"LDK", makeLDKWrapper(LDKWeight), true},
				{"LND", makeWeightFuncWrapper(LNDWeight), true},
			} {
				modelArg := model
				if !testCase.UsesProbabilityModel {
					modelArg = "N/A (capacity bias)"
				}

				success := false
				var finalPath []Channel
				var pathPe float64
				var err error
				var retries int

				for attempt := 0; attempt < maxRetries; attempt++ {
					finalPath, pathPe, err = testCase.Func(graph, src, dst, amount, modelArg)
					if err == nil && len(finalPath) > 0 {
						success = true
						retries = attempt + 1
						break
					}
				}

				res := Result{
					Source:          src,
					Destination:     dst,
					WeightFunction:  testCase.Name,
					ProbabilityModel: string(modelArg),
					Amount:          amount,
					Success:         success,
					Retries:         retries,
					PathPe:          pathPe,
				}

				if len(finalPath) > 0 {
					totalFee := 0.0
					totalHopCost := 0.0
					pathStr := ""
					for _, ch := range finalPath {
						totalFee += ch.Fee
						totalHopCost += ch.HopCost
						pathStr += fmt.Sprintf("%s -> ", ch.FromNode)
					}
					pathStr += dst
					res.PathLength = len(finalPath)
					res.TotalFee = totalFee
					res.TotalHopCost = totalHopCost
					res.Path = pathStr
				}

				results = append(results, res)
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

    numTests := 50
    ComprehensiveTest(graph, numTests)
    fmt.Println("Tests completed")
}
