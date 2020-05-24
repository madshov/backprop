package main

import (
	"fmt"
	"math"
)

var numLayers = 3
var numUnits = []int{2, 2, 2}

func main() {
	var net *Net
	var minTestErr = math.Inf(0)

	net = NewNet(0.9, 0.25, 1.0)
	net.initializeWeights()
	fmt.Printf("%+v", net)

	for {
		net.train(10)
		err := net.test()
		if err < minTestErr {
			minTestErr = err
			net.saveWeights()
		} else {
			if err > 1.2*minTestErr {
				net.restoreWeights()
				break
			}
		}
	}

	net.test()
	net.evaluate()
}
