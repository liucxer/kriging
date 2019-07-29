package kriging

import "math"

func fill_diagonal(a [][]float64, sample float64) {
	for i := 0; i < len(a); i++ {
		a[i][i] = sample
	}
}

func euclideanDistince(x [2]float64, y [2]float64) float64 {
	return math.Sqrt(math.Pow((math.Abs(y[0]-x[0])), 2) + math.Pow((math.Abs(y[1]-x[1])), 2))
}

func sqeuclideanDistince(x [2]float64, y [2]float64) float64 {
	return math.Pow((math.Abs(y[0]-x[0])), 2) + math.Pow((math.Abs(y[1]-x[1])), 2)
}

func flatten(a [][]float64) []float64 {
	var res []float64
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[i]); j++ {
			res = append(res, a[i][j])
		}
	}
	return res
}

func Meshgrid(xpts []float64, ypts []float64) ([][]float64, [][]float64) {
	var xptsRes [][]float64
	var yptsRes [][]float64

	for i := 0; i < len(ypts); i++ {
		xptsRes = append(xptsRes, xpts)
	}

	for i := 0; i < len(ypts); i++ {
		tmp := make([]float64, len(xpts))
		for j := 0; j < len(xpts); j++ {
			tmp[j] = ypts[i]
		}

		yptsRes = append(yptsRes, tmp)
	}

	return xptsRes, yptsRes
}

func cdist(x PointList, y PointList) [][]float64 {
	var res [][]float64
	for i := 0; i < len(x); i++ {
		xTmp := []float64{}
		for j := 0; j < len(y); j++ {
			yTmp := euclideanDistince(x[i], y[j])
			xTmp = append(xTmp, yTmp)
		}
		res = append(res, xTmp)
	}

	return res
}

func concatenate(x []float64, y []float64) PointList {
	xy := PointList{}
	for i := 0; i < len(x); i++ {
		tmp := [2]float64{}
		tmp[0] = x[i]
		tmp[1] = y[i]
		xy = append(xy, tmp)
	}
	return xy
}

func Dot(m [][]float64, n []float64) []float64 {
	length := len(n)
	res := make([]float64, length)

	sum := float64(0)
	for i := 0; i < length; i++ {
		sum = float64(0)
		for j := 0; j < length; j++ {
			sum += m[i][j] * n[j]
		}
		res[i] = sum
	}

	return res
}
