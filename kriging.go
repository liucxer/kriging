package kriging

import (
	"math"
	"sort"
)

type Kriging struct {
	A      float64
	K      []float64
	M      []float64
	model  krigingModel
	n      int
	nugget float64
	rangex float64
	sill   float64
	t      []float64
	x      []float64
	y      []float64
}

type krigingModel func(float64, float64, float64, float64, float64) float64

type DistanceList [][2]float64

func (t DistanceList) Len() int {
	return len(t)
}

func (t DistanceList) Less(i, j int) bool {
	return t[i][0] < t[j][0]
}

func (t DistanceList) Swap(i, j int) {
	tmp := t[i]
	t[i] = t[j]
	t[j] = tmp
}

func (Kriging) Train(t, x, y []float64, model string, sigma2 float64, alpha float64) Kriging {
	var variogram Kriging
	variogram.t = t
	variogram.x = x
	variogram.y = y
	variogram.nugget = 0
	variogram.rangex = 0
	variogram.sill = 0
	variogram.A = float64(1) / float64(3)
	variogram.n = 0

	switch model {
	case "gaussian":
		variogram.model = func(h, nugget, rangex, sill, A float64) float64 {
			return nugget + ((sill-nugget)/rangex)*
				(1.0-math.Exp(-(1.0/A)*math.Pow(h/rangex, 2)))
		}
		break
	case "exponential":
		variogram.model = func(h, nugget, rangex, sill, A float64) float64 {
			return nugget + ((sill-nugget)/rangex)*
				(1.0-math.Exp(-(1.0/A)*(h/rangex)))
		}
		break
	case "spherical":
		variogram.model = func(h, nugget, rangex, sill, A float64) float64 {
			if h > rangex {
				return nugget + (sill-nugget)/rangex
			} else {
				return nugget + ((sill-nugget)/rangex)*
					(1.5*(h/rangex)-0.5*math.Pow(h/rangex, 3))
			}
		}
		break
	}

	// Lag distance/semivariance
	var i, j, k, l, n int
	n = len(t)

	var distance DistanceList
	distance = make([][2]float64, (n*n-n)/2)

	i = 0
	k = 0
	for ; i < n; i++ {
		for j = 0; j < i; {
			distance[k] = [2]float64{}
			distance[k][0] = math.Pow(
				math.Pow(x[i]-x[j], 2)+
					math.Pow(y[i]-y[j], 2), 0.5)
			distance[k][1] = math.Abs(t[i] - t[j])
			j++
			k++
		}
	}
	sort.Sort(distance)
	variogram.rangex = distance[(n*n-n)/2-1][0]

	// Bin lag distance
	var lags int
	if ((n*n - n) / 2) > 30 {
		lags = 30
	} else {
		lags = (n*n - n) / 2
	}

	tolerance := variogram.rangex / float64(lags)

	lag := make([]float64, lags)
	semi := make([]float64, lags)
	if lags < 30 {
		for l = 0; l < lags; l++ {
			lag[l] = distance[l][0]
			semi[l] = distance[l][1]
		}
	} else {
		i = 0
		j = 0
		k = 0
		l = 0
		for i < lags && j < ((n*n-n)/2) {
			for {
				if distance[j][0] > (float64(i+1) * tolerance) {
					break
				}
				lag[l] += distance[j][0]
				semi[l] += distance[j][1]
				j++
				k++
				if j >= ((n*n - n) / 2) {
					break
				}
			}

			if k > 0 {
				lag[l] = lag[l] / float64(k)
				semi[l] = semi[l] / float64(k)
				l++
			}
			i++
			k = 0
		}
		if l < 2 {
			return variogram
		}
	}

	// Feature transformation
	n = l
	variogram.rangex = lag[n-1] - lag[0]
	X := make([]float64, 2*n)
	for i := 0; i < len(X); i++ {
		X[i] = 1
	}
	Y := make([]float64, n)
	var A = variogram.A
	for i = 0; i < n; i++ {
		switch model {
		case "gaussian":
			X[i*2+1] = 1.0 - math.Exp(-(1.0/A)*math.Pow(lag[i]/variogram.rangex, 2))
			break
		case "exponential":
			X[i*2+1] = 1.0 - math.Exp(-(1.0/A)*lag[i]/variogram.rangex)
			break
		case "spherical":
			X[i*2+1] = 1.5*(lag[i]/variogram.rangex) - 0.5*math.Pow(lag[i]/variogram.rangex, 3)
			break
		}
		Y[i] = semi[i]
	}

	// Least squares
	var Xt = krigingMatrixTranspose(X, n, 2)
	var Z = krigingMatrixMultiply(Xt, X, 2, n, 2)
	Z = krigingMatrixAdd(Z, krigingMatrixDiag(float64(1)/alpha, 2), 2, 2)
	var cloneZ = Z
	if krigingMatrixChol(Z, 2) {
		krigingMatrixChol2inv(Z, 2)
	} else {
		krigingMatrixSolve(cloneZ, 2)
		Z = cloneZ
	}
	var W = krigingMatrixMultiply(krigingMatrixMultiply(Z, Xt, 2, 2, n), Y, 2, n, 1)

	// Variogram parameters
	variogram.nugget = W[0]
	variogram.sill = W[1]*variogram.rangex + variogram.nugget
	variogram.n = len(x)

	// Gram matrix with prior
	n = len(x)
	K := make([]float64, n*n)
	for i = 0; i < n; i++ {
		for j = 0; j < i; j++ {
			K[i*n+j] = variogram.model(math.Pow(math.Pow(x[i]-x[j], 2)+
				math.Pow(y[i]-y[j], 2), 0.5),
				variogram.nugget,
				variogram.rangex,
				variogram.sill,
				variogram.A)
			K[j*n+i] = K[i*n+j]
		}
		K[i*n+i] = variogram.model(0, variogram.nugget,
			variogram.rangex,
			variogram.sill,
			variogram.A)
	}

	// Inverse penalized Gram matrix projected to target vector
	var C = krigingMatrixAdd(K, krigingMatrixDiag(sigma2, n), n, n)
	var cloneC = C
	if krigingMatrixChol(C, n) {
		krigingMatrixChol2inv(C, n)
	} else {
		krigingMatrixSolve(cloneC, n)
		C = cloneC
	}

	// Copy unprojected inverted matrix as K
	K = C
	M := krigingMatrixMultiply(C, t, n, n, 1)
	variogram.K = K
	variogram.M = M

	return variogram
}

func (variogram Kriging) Predict(x, y float64) float64 {
	k := make([]float64, variogram.n)
	for i := 0; i < variogram.n; i++ {
		k[i] = variogram.model(math.Pow(math.Pow(x-variogram.x[i], 2)+
			math.Pow(y-variogram.y[i], 2), 0.5),
			variogram.nugget, variogram.rangex,
			variogram.sill, variogram.A)
	}

	return krigingMatrixMultiply(k, variogram.M, 1, variogram.n, 1)[0]
}

// 矩阵颠倒，横向矩阵变成纵向矩阵
func krigingMatrixTranspose(X []float64, n, m int) []float64 {
	Z := make([]float64, m*n)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			Z[j*n+i] = X[i*m+j]
		}
	}

	return Z
}

// 矩阵相乘, 横向矩阵*纵向矩阵
func krigingMatrixMultiply(X, Y []float64, n, m, p int) []float64 {
	Z := make([]float64, n*p)
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			Z[i*p+j] = 0
			for k := 0; k < m; k++ {
				Z[i*p+j] += X[i*m+k] * Y[k*p+j]
			}
		}
	}
	return Z
}

func krigingMatrixAdd(X, Y []float64, n, m int) []float64 {
	Z := make([]float64, n*m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			Z[i*m+j] = X[i*m+j] + Y[i*m+j]
		}
	}

	return Z
}

func krigingMatrixDiag(c float64, n int) []float64 {
	Z := make([]float64, n*n)
	for i := 0; i < n; i++ {
		Z[i*n+i] = c
	}
	return Z
}

func krigingMatrixChol(X []float64, n int) bool {
	p := make([]float64, n)

	for i := 0; i < n; i++ {
		p[i] = X[i*n+i]
	}

	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			p[i] -= X[i*n+j] * X[i*n+j]
		}
		if p[i] <= 0 {
			return false
		}
		p[i] = math.Sqrt(p[i])
		for j := i + 1; j < n; j++ {
			for k := 0; k < i; k++ {
				X[j*n+i] -= X[j*n+k] * X[i*n+k]
				X[j*n+i] /= p[i]
			}
		}
	}

	for i := 0; i < n; i++ {
		X[i*n+i] = p[i]
	}
	return true
}

func krigingMatrixChol2inv(X []float64, n int) {
	var i, j, k int
	var sum float64

	for i = 0; i < n; i++ {
		X[i*n+i] = 1 / X[i*n+i]
		for j = i + 1; j < n; j++ {
			sum = 0
			for k = i; k < j; k++ {
				sum -= X[j*n+k] * X[k*n+i]
			}
			X[j*n+i] = sum / X[j*n+j]
		}
	}

	for i = 0; i < n; i++ {
		for j = i + 1; j < n; j++ {
			X[i*n+j] = 0
		}
	}

	for i = 0; i < n; i++ {
		X[i*n+i] *= X[i*n+i]
		for k = i + 1; k < n; k++ {
			X[i*n+i] += X[k*n+i] * X[k*n+i]
		}

		for j = i + 1; j < n; j++ {
			for k = j; k < n; k++ {
				X[i*n+j] += X[k*n+i] * X[k*n+j]
			}
		}
	}

	for i = 0; i < n; i++ {
		for j = 0; j < i; j++ {
			X[i*n+j] = X[j*n+i]
		}
	}
}

func krigingMatrixSolve(X []float64, n int) bool {
	var m = n
	var b = make([]float64, n*n)
	var indxc = make([]int, n)
	var indxr = make([]int, n)
	var ipiv = make([]int, n)
	var i, icol, irow, j, k, l, ll int
	var big, dum, pivinv, temp float64

	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			if i == j {
				b[i*n+j] = 1
			} else {
				b[i*n+j] = 0
			}
		}
	}

	for j = 0; j < n; j++ {
		ipiv[j] = 0
	}

	for i = 0; i < n; i++ {
		big = 0
		for j = 0; j < n; j++ {
			if ipiv[j] != 1 {
				for k = 0; k < n; k++ {
					if ipiv[k] == 0 {
						if math.Abs(X[j*n+k]) >= big {
							big = math.Abs(X[j*n+k])
							irow = j
							icol = k
						}
					}
				}
			}
		}
		ipiv[icol]++
		if irow != icol {
			for l = 0; l < n; l++ {
				temp = X[irow*n+l]
				X[irow*n+l] = X[icol*n+l]
				X[icol*n+l] = temp
			}
			for l = 0; l < m; l++ {
				temp = b[irow*n+l]
				b[irow*n+l] = b[icol*n+l]
				b[icol*n+l] = temp
			}
		}
		indxr[i] = irow
		indxc[i] = icol

		if X[icol*n+icol] == 0 {
			return false
		} // Singular

		pivinv = 1 / X[icol*n+icol]
		X[icol*n+icol] = 1
		for l = 0; l < n; l++ {
			X[icol*n+l] *= pivinv
		}
		for l = 0; l < m; l++ {
			b[icol*n+l] *= pivinv
		}

		for ll = 0; ll < n; ll++ {
			if ll != icol {
				dum = X[ll*n+icol]
				X[ll*n+icol] = 0
				for l = 0; l < n; l++ {
					X[ll*n+l] -= X[icol*n+l] * dum
				}
				for l = 0; l < m; l++ {
					b[ll*n+l] -= b[icol*n+l] * dum
				}
			}
		}
	}
	for l = n - 1; l >= 0; l-- {
		if indxr[l] != indxc[l] {
			for k = 0; k < n; k++ {
				temp = X[k*n+indxr[l]]
				X[k*n+indxr[l]] = X[k*n+indxc[l]]
				X[k*n+indxc[l]] = temp
			}
		}
	}

	return true
}
