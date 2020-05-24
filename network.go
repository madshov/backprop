package main

import (
	"fmt"
)

type Net struct {
	Layers []*Layer
	α      float64
	η      float64
	gain   float64
	Error  float64
}

func NewNet(α, η, gain float64) *Net {
	ls := make([]*Layer, numLayers)

	for i := 0; i < numLayers; i++ {
		ls[i] = NewLayer(i-1, i)
	}

	return &Net{
		Layers: ls,
		α:      α,
		η:      η,
		gain:   gain,
	}
}

func (n *Net) initializeWeights() {
	for i := 1; i < numLayers; i++ {
		cur := n.Layers[i]
		prev := n.Layers[i-1]
		cur.initializeWeight(prev)
	}
}

func (n *Net) saveWeights() {
	for i := 1; i < numLayers; i++ {
		cur := n.Layers[i]
		prev := n.Layers[i-1]
		cur.saveWeight(prev)
	}
}

func (n *Net) restoreWeights() {
	for i := 1; i < numLayers; i++ {
		cur := n.Layers[i]
		prev := n.Layers[i-1]
		cur.restoreWeight(prev)
	}
}

func (n *Net) adjustWeights() {
	for i := 1; i < numLayers; i++ {
		cur := n.Layers[i]
		prev := n.Layers[i-1]
		cur.adjustWeight(prev, n.α, n.η)
	}
}

func (n *Net) inputLayer() *Layer {
	return n.Layers[0]
}

func (n *Net) outputLayer() *Layer {
	return n.Layers[numLayers-1]
}

func (n *Net) setInput(in []float64) {
	n.inputLayer().setInput(in)
}

func (n *Net) getOutput() []float64 {
	return n.outputLayer().getOutput()
}

func (n *Net) computeErrorTotal(targets []float64) float64 {
	return n.outputLayer().computeError(targets, n.gain)
}

func (n *Net) propagate() {
	for i := 0; i < numLayers-1; i++ {
		cur := n.Layers[i]
		next := n.Layers[i+1]
		cur.propagate(next, n.gain)
	}
}

func (n *Net) backPropagate() {
	for i := numLayers - 1; i > 1; i-- {
		cur := n.Layers[i]
		prev := n.Layers[i-1]
		cur.backPropagate(prev, n.gain)
	}
}

func (n *Net) simulate(input, targets []float64, train bool) []float64 {
	n.setInput(input)
	n.propagate()

	o := n.getOutput()
	n.Error = n.computeErrorTotal(targets)

	if train {
		n.backPropagate()
		n.adjustWeights()
	}

	return o
}

func (n *Net) train(epochs int) {
	for i := 0; i < epochs; i++ {
		var data []float64
		n.simulate(data, data, true)
	}
}

func (n *Net) test() float64 {
	minErr := 0.0

	for i := 0; i < 10; i++ {
		var data []float64
		n.simulate(data, data, false)
		minErr += n.Error
	}

	return minErr
}

func (n *Net) evaluate() {
	for i := 0; i < 10; i++ {
		//var data []float64
		//output := net.simulate(data, data, false)
	}
}

func (n *Net) print() {
	for i := 1; i < numLayers; i++ {
		fmt.Printf("Layer %d:\n", i)
		for j := 1; j < n.Layers[i].Units; j++ {
			fmt.Printf("\t Unit %d:\n", j)
			for k := 0; k < n.Layers[i-1].Units; k++ {
				fmt.Printf("%f ", n.Layers[i].Weights[j][k])
			}
			fmt.Println()
		}
		fmt.Println()
	}
}
