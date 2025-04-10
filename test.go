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
	PeCache		float64
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
var bimodalPeTable = make(map[int]map[int]float64)

func PrecomputeBimodalPeTable() {
    for capacity := 1000; capacity <= 20_000_000; capacity += 1000 {
        innerMap := make(map[int]float64)
        for ratio := 1; ratio <= 100; ratio++ {
            amt := float64(ratio) / 100.0 * float64(capacity)
            ch := Channel{Capacity: float64(capacity)}
            pe := EstimateBimodalSuccessProbability(ch, amt)
            innerMap[ratio] = pe
        }
        bimodalPeTable[capacity] = innerMap
    }
}

func EstimateBimodalSuccessProbability(ch Channel, amount float64) float64 {
	if amount >= ch.Capacity || ch.Capacity == 0 {
		return 1e-9
	}

	s := ch.Capacity / 10.0
	pdf := func(x float64) float64 {
		return math.Exp(-x/s) + math.Exp((x - ch.Capacity)/s)
	}

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

	numerator := integrate(pdf, amount, ch.Capacity, 200)
	denominator := integrate(pdf, 0, ch.Capacity, 200)

	if denominator == 0 {
		return 1e-9
	}

	pe := numerator / denominator
	// fmt.Printf("  DEBUG: Pe = %.9f from %s to %s\n", pe, ch.FromNode, ch.ToNode)

	return math.Max(pe, 1e-9)
}

func LookupBimodalSuccessProbability(ch Channel, amount float64) float64 {
    capKey := int(ch.Capacity / 1000) * 1000
    ratio := int((amount / ch.Capacity) * 100)

    if capMap, ok := bimodalPeTable[capKey]; ok {
        if pe, ok2 := capMap[ratio]; ok2 {
            return math.Max(pe, 1e-9)
        }
    }

    // fallback if not found
    return EstimateBimodalSuccessProbability(ch, amount)
}


// estimate success probability using uniform function
// pe = (cap - amt) / cap
func EstimateUniformSuccessProbability(ch Channel, amount float64) float64 {
	if ch.Capacity == 0 || amount >= ch.Capacity {
		return 1e-9
	}
	return math.Max((ch.Capacity - amount) / ch.Capacity, 1e-9)
}


type ProbabilityModel string

const (
    Bimodal ProbabilityModel = "bimodal"
    Uniform ProbabilityModel = "uniform"
)


// LND weight function
// add constraints in weight functions and dijkstra, should check intermediary in dijkstra for lnd and ldk, and at end for eclair and cln
func LNDWeight(ch Channel, currentProb float64, model ProbabilityModel, amount float64) (float64, float64) {
    var Pe float64
    if model == Bimodal {
        Pe = LookupBimodalSuccessProbability(ch, amount)
    } else {
        Pe = EstimateUniformSuccessProbability(ch, amount)
    }

    additive := ch.Fee + (amount * ch.CLTV * 15e-9)
    penalty := 100.0 + amount * 1000.0
    penalty /= 1000 // msat to sats

    multiplicative := penalty / (math.Max(currentProb * Pe, 1e-9))

    return additive, multiplicative
}

func EclairWeight(ch Channel, model ProbabilityModel, amount float64) float64 {
    var Pe float64
    if model == Bimodal {
        Pe = LookupBimodalSuccessProbability(ch, amount)
    } else {
        Pe = EstimateUniformSuccessProbability(ch, amount)
    }
    
    lockedFundsRisk := 1e-8 // per paper
    riskCost := amount * ch.CLTV * lockedFundsRisk

	logPe := math.Log(math.Max(Pe, 1e-9))
	penalty := ((2000 + amount*500) / 1000) * logPe
	total := (ch.Fee + ch.HopCost + riskCost) - penalty
    
    // return (ch.Fee + ch.HopCost + riskCost) - ((2000+ch.Amt*500)/1000)*math.Log(Pe) // /1000 to convert from msat to satoshi?
	return total
}

// CLN weight function as per the paper
// Note: CLN does NOT use success probability like other implementations
func CLNWeight(ch Channel, amount float64) float64 {
    blockPerYear := 52596.0
    riskFactor := 10.0
	// this capacityBias is success prob in CLN
	estimateSuccessProb := (ch.Capacity + 1 - amount) / (ch.Capacity + 1)
    capacityBias := -math.Log(estimateSuccessProb)
    
    return (ch.Fee + (amount * ch.CLTV * riskFactor) / (blockPerYear * 100) + 1) * (capacityBias + 1)
}

// LDK weight function as per the paper
func LDKWeight(ch Channel, model ProbabilityModel, amount float64) float64 {
    // Calculate pathHtlcMin
    pathHtlcMin := ch.HtlcMin*(1+ch.FeeRate/1000000) + ch.FeeBase
    
    // Calculate base penalty
    penaltyBase := 500.0 // msat
    baseMultiplier := 8192.0 // msat
    basePenalty := penaltyBase + (baseMultiplier * amount) / math.Pow(2, 30)
    
    // Calculate anti-probing penalty
    antiProbingPenalty := 0.0
    if ch.HtlcMax >= ch.Capacity/2 {
        antiProbingPenalty = 250.0 // msat
    }
    
    // Calculate liquidity penalty
    LM := 30000.0 // msat
    amtMultiplier := (192.0 * amount) / math.Pow(2, 20)
    var liquidityPenalty float64
    
    // Using uniform distribution for success probability
	// not using exponential in bimodal; instead is polynomial
    var Pe float64
    if model == Bimodal {
        Pe = LookupBimodalSuccessProbability(ch, amount)
    } else {
        Pe = EstimateUniformSuccessProbability(ch, amount)
    }
    liquidityPenalty = -math.Log10(Pe) * (LM + amtMultiplier)
    
    // Calculate historic penalty (simplified - paper mentions historical data)
    HM := 10000.0 // msat
    historicMultiplier := (64.0 * amount) / math.Pow(2, 20)
    historicPenalty := -math.Log10(Pe) * (HM + historicMultiplier)
    
    // Total penalty
    penalty := basePenalty + antiProbingPenalty + liquidityPenalty + historicPenalty
    
    // Convert penalties from msat to satoshis
    penalty /= 1000
    
    return math.Max(ch.Fee, pathHtlcMin) + penalty
}

func CombinedWeight(ch Channel, amount float64, model ProbabilityModel) (float64, float64) {
	var Pe float64
	if model == Bimodal {
		Pe = LookupBimodalSuccessProbability(ch, amount)
	} else {
		Pe = EstimateUniformSuccessProbability(ch, amount)
	}
	Pe = math.Max(Pe, 1e-9)

	riskCost := amount * ch.CLTV * 1.8e-8
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
func FindBestRoute(graph *Graph, start, end string, amount float64,
    weightFunc func(Channel, float64) (float64, float64),
    useMultiplicative bool,
    model ProbabilityModel) ([]Channel, float64, error) {

    pq := &PriorityQueue{}
    heap.Init(pq)

    dist := make(map[string]float64)
    prev := make(map[string]Channel)
    path := make(map[string][]Channel)
    visited := make(map[string]bool)

    for node := range graph.Nodes {
        dist[node] = math.Inf(1)
    }
    dist[start] = 0.0
    path[start] = []Channel{}

    heap.Push(pq, &Item{
        node:         start,
        additiveCost: 0.0,
        path:         []Channel{},
    })

    for pq.Len() > 0 {
        current := heap.Pop(pq).(*Item)
        currentNode := current.node

        if visited[currentNode] {
            continue
        }
        visited[currentNode] = true

        if currentNode == end {
            finalPath := path[end]
            pathPe := computePathSuccessProbability(finalPath) // Pe now cached
            return finalPath, pathPe, nil
        }

        for _, ch := range graph.Nodes[currentNode] {
            if ch.Capacity < amount {
                continue
            }

            additive, multiplicative := weightFunc(ch, amount)
			pe := multiplicative
			if !useMultiplicative {
				ch.PeCache = pe // Save just Pe
			} else {
				ch.PeCache = 1.0 // LND handles reliability in the cost
			}
			cost := additive
			if useMultiplicative {
				cost += multiplicative
			}

            newDist := dist[currentNode] + cost

            if newDist < dist[ch.ToNode] {
                dist[ch.ToNode] = newDist
                prev[ch.ToNode] = ch

                newPath := make([]Channel, len(path[currentNode]))
                copy(newPath, path[currentNode])
                newPath = append(newPath, ch)
                path[ch.ToNode] = newPath

                heap.Push(pq, &Item{
                    node:         ch.ToNode,
                    additiveCost: newDist,
                    path:         newPath,
                })
            }
        }
    }

    return nil, 0.0, fmt.Errorf("no path found")
}

// helper function for dijkstra
func computePathSuccessProbability(path []Channel) float64 {
    pathPe := 1.0
    for _, ch := range path {
        pe := ch.PeCache
        if pe <= 0.0 {
            return 0.0
        }
        pathPe *= pe
    }
    return pathPe
}

func makeWeightFuncWrapper(
    weightFunc func(Channel, float64, ProbabilityModel) (float64, float64),
    useMultiplicative bool,
) func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, float64, error) {
        return FindBestRoute(graph, start, end, amount, func(ch Channel, _ float64) (float64, float64) {
            return weightFunc(ch, amount, model)
        }, useMultiplicative, model)
    }
}

func makeEclairWrapper(weightFunc func(Channel, ProbabilityModel, float64) float64) func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, float64, error) {
        return FindBestRoute(graph, start, end, amount, func(ch Channel, _ float64) (float64, float64) {
            var pe float64
            if model == Bimodal {
                pe = LookupBimodalSuccessProbability(ch, amount)
            } else {
                pe = EstimateUniformSuccessProbability(ch, amount)
            }
            return weightFunc(ch, model, amount), pe
        }, false, model)
    }
}

func makeLDKWrapper(weightFunc func(Channel, ProbabilityModel, float64) float64) func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, float64, error) {
        return FindBestRoute(graph, start, end, amount, func(ch Channel, _ float64) (float64, float64) {
            var pe float64
            if model == Bimodal {
                pe = LookupBimodalSuccessProbability(ch, amount)
            } else {
                pe = EstimateUniformSuccessProbability(ch, amount)
            }
            return weightFunc(ch, model, amount), pe
        }, false, model)
    }
}

func makeCLNWrapper() func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error) {
    return func(graph *Graph, start, end string, amount float64, model ProbabilityModel) ([]Channel, float64, error) {
        return FindBestRoute(graph, start, end, amount, func(ch Channel, _ float64) (float64, float64) {
            return CLNWeight(ch, amount), 1.0
        }, false, model)
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

        maxSrcAmt := maxSendable(graph, src)
        maxDstAmt := maxReceivable(graph, dst)
        feasibleAmt := math.Min(maxSrcAmt, maxDstAmt)
        if feasibleAmt < 1000 {
            continue
        }

		// stay within a safe margin (90% of capacity):
		amount := 1000 + r.Float64()*((feasibleAmt * 0.9) - 1000)

        for _, model := range []ProbabilityModel{Bimodal, Uniform} {
            for _, testCase := range []struct {
                Name                 string
                Func                 func(*Graph, string, string, float64, ProbabilityModel) ([]Channel, float64, error)
                UsesProbabilityModel bool
            }{
                {"Combined", makeWeightFuncWrapper(CombinedWeight, true), true},
                {"Eclair", makeEclairWrapper(EclairWeight), true},
                {"CLN", makeCLNWrapper(), false},
                {"LDK", makeLDKWrapper(LDKWeight), true},
                {"LND", makeWeightFuncWrapper(func(ch Channel, amt float64, m ProbabilityModel) (float64, float64) {
                    return LNDWeight(ch, 1.0, m, amt)
                }, true), true},
            } {
                modelUsed := model
                if !testCase.UsesProbabilityModel {
                    modelUsed = "N/A (capacity bias)"
                }

                var path []Channel
                var pathPe float64
                var err error
                success := false
                retries := 0

                for attempt := 0; attempt < maxRetries; attempt++ {
                    path, pathPe, err = testCase.Func(graph, src, dst, amount, model)
                    if err == nil && len(path) > 0 {
                        success = true
                        retries = attempt + 1
                        break
                    }
                }

                res := Result{
                    Source:           src,
                    Destination:      dst,
                    WeightFunction:   testCase.Name,
                    ProbabilityModel: string(modelUsed),
                    Amount:           amount,
                    Success:          success,
                    Retries:          retries,
                    PathPe:           pathPe,
                }

                if success {
                    totalFee := 0.0
                    totalHopCost := 0.0
                    pathStr := ""
                    for _, ch := range path {
                        totalFee += ch.Fee
                        totalHopCost += ch.HopCost
                        pathStr += fmt.Sprintf("%s -> ", ch.FromNode)
                    }
                    pathStr += dst

                    res.TotalFee = totalFee
                    res.TotalHopCost = totalHopCost
                    res.Path = pathStr
                    res.PathLength = len(path)
                }

                results = append(results, res)
            }
        }
    }

    SaveComprehensiveResults(results, "comprehensive_results.csv")
}


func main() {
	fmt.Println("Pre-computing Bimodal Pe Lookup Table...")
	PrecomputeBimodalPeTable()

    fmt.Println("Loading network data...")
    graph, err := LoadCSV("LN_snapshot.csv")
    if err != nil {
        fmt.Println("Error loading network data:", err)
        return
    }

    fmt.Printf("Loaded network with %d nodes\n", len(graph.Nodes))
    fmt.Println("Running tests...")

    numTests := 25
    ComprehensiveTest(graph, numTests)
    fmt.Println("Tests completed")
}
