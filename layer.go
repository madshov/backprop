package main

import (
	"math"
	"math/rand"
)

type Layer struct {
	Units    int
	Output   []float64
	Error    []float64
	Weights  [][]float64
	sWeights [][]float64
	ΔWeights [][]float64
}

func NewLayer(prev, no int) *Layer {
	var (
		o, e      []float64
		w, sw, Δw [][]float64
	)

	units := numUnits[no]
	pUnits := numUnits[prev]

	if prev >= 0 {
		w = make([][]float64, units)
		sw = make([][]float64, units)
		Δw = make([][]float64, units)

		for i := 0; i < len(w); i++ {
			w[i] = make([]float64, pUnits)
			sw[i] = make([]float64, pUnits)
			Δw[i] = make([]float64, pUnits)
		}
	}

	o = make([]float64, units)
	e = make([]float64, units)

	return &Layer{
		Units:    units,
		Output:   o,
		Error:    e,
		Weights:  w,
		sWeights: sw,
		ΔWeights: Δw,
	}
}

func (l *Layer) initializeWeight(prev *Layer) {
	for j := 1; j < l.Units; j++ {
		for k := 0; k < prev.Units; k++ {
			l.Weights[j][k] = -0.5 + rand.Float64()
		}
	}
}

func (l *Layer) saveWeight(prev *Layer) {
	for j := 1; j < l.Units; j++ {
		for k := 0; k < prev.Units; k++ {
			l.sWeights[j][k] = l.Weights[j][k]
		}
	}
}

func (l *Layer) restoreWeight(prev *Layer) {
	for j := 1; j < l.Units; j++ {
		for k := 0; k < prev.Units; k++ {
			l.Weights[j][k] = l.sWeights[j][k]
		}
	}
}

func (l *Layer) adjustWeight(prev *Layer, α, η float64) {
	for j := 1; j < l.Units; j++ {
		for k := 0; k < prev.Units; k++ {
			o := prev.Output[k]
			e := l.Error[j]
			Δw := l.ΔWeights[j][k]

			l.Weights[j][k] = η*e*o + α*Δw
			l.ΔWeights[j][k] = η * e * o
		}
	}
}

func (l *Layer) setInput(in []float64) {
	for i := 1; i <= l.Units; i++ {
		l.Output[i] = in[i-1]
	}
}

func (l *Layer) getOutput() []float64 {
	out := make([]float64, 0)

	for i := 1; i < l.Units; i++ {
		out[i-1] = l.Output[i]
	}

	return out
}

func (l *Layer) computeError(targets []float64, gain float64) float64 {
	sum := 0.0

	for i := 1; i < l.Units; i++ {
		o := l.Output[i]
		e := targets[i-1] - o
		l.Error[i] = gain * o * (1 - o) * e
		sum += 0.5 * math.Sqrt(e)
	}

	return sum
}

func (l *Layer) propagate(next *Layer, gain float64) {
	for i := 1; i <= next.Units; i++ {
		sum := 0.0
		for j := 0; j <= l.Units; j++ {
			sum += next.Weights[i][j] * l.Output[j]
		}

		next.Output[i] = 1 / (1 + math.Exp(-1*gain*sum))
	}
}

func (l *Layer) backPropagate(prev *Layer, gain float64) {
	for i := 1; i <= prev.Units; i++ {
		output := prev.Output[i]
		error := 0.0
		for j := 1; j <= l.Units; j++ {
			error += l.Weights[j][i] * l.Error[j]
		}

		prev.Error[i] = gain * output * (1 - output) * error
	}
}
