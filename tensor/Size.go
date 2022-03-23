package tensor

type Shape []int
type Stride []int

func (s Shape) getLength() int {
	length, _ := s.getLengthAndStride()
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
	_, stride := s.getLengthAndStride()
	return stride
}

func (s Shape) getLengthAndStride() (int, Stride) {
	stride := make([]int, len(s))
	_stride := 1
	for i, length := range s {
		stride[len(s)-i-1] = _stride
		_stride *= length
	}
	return _stride, stride
}
