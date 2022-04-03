package tensor

import (
	"GoDLearn/util"
	"errors"
	"fmt"
	"reflect"
)

func (t *Tensor[T]) operatableWith(other *Tensor[T]) error {
	if !t.shape.equalTo(other.shape) {
		return errors.New("shape")
	}
	if inputType, otherType := reflect.TypeOf(t.data), reflect.TypeOf(other.data); inputType != otherType {
		return errors.New("type")
	}
	return nil
}

func (t *Tensor[T]) multipliableWith(other *Tensor[T]) error {
	if t.dim <= 2 && other.dim <= 2 {
		if t.shape[t.dim-1] != other.shape[0] {
			return errors.New("dimesion-size")
		}
		return nil
	}

	tShape := util.Reverse(t.getFirstShapes(-2))
	otherShape := util.Reverse(other.getFirstShapes(-2))

	for i := 0; i < util.Min(len(tShape), len(otherShape)); i++ {
		if tShape[i] != otherShape[i] && tShape[i] != 1 && otherShape[i] != 1 {
			return errors.New("shape-mismatch")
		}
	}
	return nil
}

func (t *Tensor[T]) AddScaled(other *Tensor[T], alpha T) *Tensor[T] {
	err := t.operatableWith(other)
	if err != nil {
		switch err.Error() {
		case "shape":
			errString := fmt.Sprint("Cannot add two tensors with different shapes: ", t.shape, other.shape)
			panic(errString)
		case "type":
			errString := fmt.Sprint("Cannot add two tensors with different types: ", reflect.TypeOf(t.data[0]), reflect.TypeOf(other.data[0]))
			panic(errString)
		}
	}

	result := t.CopyShape()
	for i := range result.data {
		result.data[i] = t.data[i] + other.data[i]*alpha
	}
	return result
}

func (t *Tensor[T]) Add(other *Tensor[T]) *Tensor[T] {
	return t.AddScaled(other, 1)
}

func (t *Tensor[T]) Sub(other *Tensor[T]) *Tensor[T] {
	err := t.operatableWith(other)
	if err != nil {
		switch err.Error() {
		case "shape":
			errString := fmt.Sprint("Cannot do operations between two tensors with different shapes: ", t.shape, other.shape)
			panic(errString)
		case "type":
			errString := fmt.Sprint("Cannot do operations between two tensors with different types: ", reflect.TypeOf(t.data[0]), reflect.TypeOf(other.data[0]))
			panic(errString)
		}
	}

	result := t.CopyShape()
	for i := range result.data {
		result.data[i] = t.data[i] - other.data[i]
	}
	return result
}

func (t *Tensor[T]) MatMul(other *Tensor[T]) *Tensor[T] {
	err := t.multipliableWith(other)
	if err != nil {
		panic(err)
	}

	if t.dim == 1 {
		t.Reshape(1, -1)
	}
	if other.dim == 1 {
		other.Reshape(-1, 1)
	}
	// t: n x m, other: m x l
	firstShapes := t.getFirstShapes(-2)
	n, m := t.getLastTwoShapes()
	l := other.getLastShape()

	result := NewZeroTensor[T](append(firstShapes, n, l)...)

	// TODO: Implement broadcating

	for i := 0; i < n; i++ {
		for k := 0; k < l; k++ {
			for j := 0; j < m; j++ {
				result.data[i*l+k] += t.data[i*m+j] * other.data[j*l+k]
			}
		}
	}

	return result
}
