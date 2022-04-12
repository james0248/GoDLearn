package main

import "GoDLearn/tensor"

func main() {
	//l := nn.NewLinear[int](2, 3, nn.WithBias(true))
	//fmt.Println(l.Weight())
	t := tensor.NewZeroTensor[float32](2, 3, 4)
	t.Uniform(1, 3)
}
