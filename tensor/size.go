package tensor

import "errors"

type Shape []int
type Stride []int

func InferShape(_size int, _shape []int) (Shape, error) {
	inferredIndex := -1
	size := 1
	for i := range _shape {
		if _shape[i] == -1 {
			if inferredIndex != -1 {
				return nil, errors.New("infer-impossible")
			}
			inferredIndex = i
		}
		size *= _shape[i]
	}
	if inferredIndex == -1 {
		if _size != size {
			return nil, errors.New("size-mismatch")
		}
	} else {
		if _size%size != 0 {
			return nil, errors.New("infer-mismatch")
		}
		_shape[inferredIndex] = _size / size
	}
	return Shape(_shape), nil
}

func (s Shape) getSize() int {
	length, _ := s.getSizeAndStride()
	return length
}

func (s Shape) equalTo(other Shape) bool {
	if len(s) != len(other) {
		return false
	}
	for i := range s {
		if s[i] != other[i] {
			return false
		}
	}
	return true
}

func (s Shape) getStride() Stride {
	_, stride := s.getSizeAndStride()
	return stride
}

func (s Shape) getSizeAndStride() (int, Stride) {
	stride := make([]int, len(s))
	_stride := 1
	for i := range s {
		stride[len(s)-i-1] = _stride
		_stride *= s[len(s)-i-1]
	}
	return _stride, stride
}
