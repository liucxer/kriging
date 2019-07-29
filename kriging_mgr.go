package kriging

import (
	"math"
)

type KrigingPy struct {
	X_ORIG FloatList
	Y_ORIG FloatList
	Z      []float64

	Verbose        bool
	EnablePlotting bool

	XCENTER           float64
	YCENTER           float64
	AnisotropyScaling float64
	AnisotropyAngle   float64
	X_ADJUSTED        []float64
	Y_ADJUSTED        []float64

	CoordinatesType string
	VariogramModel  string

	VariogramFunction        VariogramFunction
	Lags                     []float64
	Semivariance             []float64
	VariogramModelParameters []float64
}

type VariogramFunction func([]float64, [][]float64) [][]float64

type FloatList struct {
	V   []float64
	Max float64
	Min float64
	Avg float64
	Sum float64
	Len int
}

func NewFloatList(v []float64) FloatList {
	var floatList FloatList

	floatList.V = v
	floatList.Max = Max(v)
	floatList.Min = Min(v)
	floatList.Sum = Sum(v)
	floatList.Avg = Avg(v)
	floatList.Len = len(v)
	return floatList
}

func Sum(l []float64) float64 {
	sum := float64(0)
	for i := 0; i < len(l); i++ {
		sum = sum + l[i]
	}
	return sum
}

func Avg(l []float64) float64 {
	sum := float64(0)
	for i := 0; i < len(l); i++ {
		sum = sum + l[i]
	}
	return sum / float64(len(l))
}

func Max(l []float64) float64 {
	max := float64(0)
	for i := 0; i < len(l); i++ {
		if l[i] > max {
			max = l[i]
		}
	}
	return max
}

func Min(l []float64) float64 {
	min := float64(0)
	for i := 0; i < len(l); i++ {
		if l[i] < min || min == 0 {
			min = l[i]
		}
	}
	return min
}

type PointList [][2]float64

func NewKrigingPy(x []float64, y []float64, z []float64, variogramModel string, variogramParameters []float64,
	variogramFunction VariogramFunction, nlags int, weight bool, anisotropyScaling float64, anisotropyAngle float64,
	verbose bool, enablePlotting bool, enableStatistics bool, coordinatesType string) KrigingPy {

	var krigingPy KrigingPy

	if variogramModel == "" {
		variogramModel = "linear"
	}

	if nlags == 0 {
		nlags = 6
	}

	if anisotropyScaling == 0 {
		anisotropyScaling = 1.0
	}

	if coordinatesType == "" {
		coordinatesType = "euclidean"
	}

	krigingPy.X_ORIG = NewFloatList(x)
	krigingPy.Y_ORIG = NewFloatList(y)
	krigingPy.Z = z
	krigingPy.Verbose = verbose
	krigingPy.EnablePlotting = enablePlotting

	if coordinatesType == "euclidean" {
		krigingPy.XCENTER = (krigingPy.X_ORIG.Max + krigingPy.X_ORIG.Min) / 2.0
		krigingPy.YCENTER = (krigingPy.Y_ORIG.Max + krigingPy.Y_ORIG.Min) / 2.0
		krigingPy.AnisotropyScaling = anisotropyScaling
		krigingPy.AnisotropyAngle = anisotropyAngle

		var pointList PointList

		for i := 0; i < len(krigingPy.X_ORIG.V); i++ {
			var point [2]float64
			point[0] = krigingPy.X_ORIG.V[i]
			point[1] = krigingPy.Y_ORIG.V[i]
			pointList = append(pointList, point)
		}
		krigingPy.X_ADJUSTED, krigingPy.Y_ADJUSTED =
			_adjust_for_anisotropy(pointList, [2]float64{krigingPy.XCENTER, krigingPy.YCENTER}, krigingPy.AnisotropyScaling, krigingPy.AnisotropyAngle)
	} else if coordinatesType == "geographic" {
		krigingPy.XCENTER = 0.0
		krigingPy.YCENTER = 0.0
		krigingPy.AnisotropyScaling = 1.0
		krigingPy.AnisotropyAngle = 0.0
		krigingPy.X_ADJUSTED = krigingPy.X_ORIG.V
		krigingPy.Y_ADJUSTED = krigingPy.Y_ORIG.V
	} else {
		panic("coordinatesType error")
	}

	krigingPy.CoordinatesType = coordinatesType
	krigingPy.VariogramModel = variogramModel
	if _, ok := variogramDictMap[variogramModel]; ok {
		krigingPy.VariogramFunction = variogramDictMap[variogramModel]
	}

	var pointList PointList

	for i := 0; i < krigingPy.X_ORIG.Len; i++ {
		var point [2]float64
		point[0] = krigingPy.X_ADJUSTED[i]
		point[1] = krigingPy.Y_ADJUSTED[i]
		pointList = append(pointList, point)
	}

	krigingPy.Lags, krigingPy.Semivariance, krigingPy.VariogramModelParameters =
		InitializeVariogramModel(pointList, krigingPy.Z, krigingPy.VariogramModel, variogramParameters,
			krigingPy.VariogramFunction, nlags, weight, krigingPy.CoordinatesType)
	return krigingPy
}

func InitializeVariogramModel(X PointList, y []float64, variogramModel string,
	variogramModelParameters []float64, variogramFunction VariogramFunction,
	nlags int, weight bool, coordinatesType string) ([]float64, []float64, []float64) {

	var d FloatList
	var g FloatList
	if coordinatesType == "euclidean" {
		d = NewFloatList(pdist(X, "euclidean"))
		var yPointList PointList
		for i := 0; i < len(y); i++ {
			var point [2]float64
			point[0] = y[i]
			yPointList = append(yPointList, point)
		}
		g = NewFloatList(pdist(yPointList, "sqeuclidean"))
		for i := 0; i < g.Len; i++ {
			g.V[i] = 0.5 * g.V[i]
		}
	} else {
		panic("coordinatesType != euclidean")
	}

	dmax := d.Max
	dmin := d.Min
	dd := (dmax - dmin) / float64(nlags)
	var bins []float64
	for i := 0; i < nlags; i++ {
		bins = append(bins, dd*float64(i))
	}
	dmax = dmax + 0.001
	bins = append(bins, dmax)

	lags := make([]float64, nlags)
	semivariance := make([]float64, nlags)
	for n := 0; n < nlags; n++ {

		var avgLagsFloat []float64
		for m := 0; m < d.Len; m++ {
			if d.V[m] > bins[n] && d.V[m] < bins[n+1] {
				avgLagsFloat = append(avgLagsFloat, d.V[m])
			}
		}

		if len(avgLagsFloat) > 0 {
			lags[n] = Avg(avgLagsFloat)
		}

		var avgSemivarianceFloat []float64
		for m := 0; m < d.Len; m++ {
			if d.V[m] > bins[n] && d.V[m] < bins[n+1] {
				avgSemivarianceFloat = append(avgSemivarianceFloat, g.V[m])
			}
		}
		if len(avgSemivarianceFloat) > 0 {
			semivariance[n] = Avg(avgSemivarianceFloat)
		}
	}

	return lags, semivariance, variogramModelParameters
}

func pdist(X PointList, metric string) []float64 {
	var res []float64
	n := len(X)
	res = make([]float64, (n*(n-1))/2)
	current := 0
	if metric == "euclidean" {
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				res[current] = euclideanDistince(X[i], X[j])
				current++
			}
		}
		return res
	} else if metric == "sqeuclidean" {
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				res[current] = sqeuclideanDistince(X[i], X[j])
				current++
			}
		}
		return res
	} else {
		panic("pdist metric")
	}
	return nil
}

var variogramDictMap = map[string]VariogramFunction{
	"linear": LinearVariogramModel,
	/*
		'power': variogram_models.power_variogram_model,
				'gaussian': variogram_models.gaussian_variogram_model,
				'spherical': variogram_models.spherical_variogram_model,
				'exponential': variogram_models.exponential_variogram_model,
				'hole-effect': variogram_models.hole_effect_variogram_model
	*/
}

func LinearVariogramModel(m []float64, d [][]float64) [][]float64 {
	slope := m[0]
	nugget := m[1]

	for i := 0; i < len(d); i++ {
		for j := 0; j < len(d[i]); j++ {
			d[i][j] = d[i][j]*slope + nugget
		}
	}
	return d
}

func LinearVariogramModelLine(m []float64, d []float64) []float64 {
	slope := m[0]
	nugget := m[1]

	for i := 0; i < len(d); i++ {
		d[i] = d[i]*slope + nugget
	}
	return d
}

const PI = 3.141592653589793

//  两个数组的点积
//           B{[5,6]
//             [7,8]}
//
// A {[1,2]  C{[19,22]
// 	  [3,4]}   [43,50]}
// A*B : 1*5 + 2*7 = 19
// 		 1*6 + 2*8 = 22
// 		 3*5 + 4*7 = 43
//       3*6 + 4*8 = 50
func MatrixDot(X [][]float64, Y [][]float64) []float64 {
	return nil
	//return Z
}

// 矩阵颠倒，横向矩阵变成纵向矩阵
// |1|0.06
// |1|0.14
// |1|0.22
// |1|0.29
// |1|0.37
// 转化成
// |1	|1	 |1   |1   |1
// |0.06|0.14|0.22|0.29|0.37
func MatrixTranspose(X [][2]float64) [2][]float64 {
	if len(X) == 0 {
		return [2][]float64{}
	}

	hight := len(X)    // 5
	width := len(X[0]) // 2

	var Z [2][]float64
	for i := 0; i < width; i++ {
		var tmp []float64
		for j := 0; j < hight; j++ {
			tmp = append(tmp, X[j][i])
		}
		Z[i] = tmp
	}

	return Z
}

func _adjust_for_anisotropy(X PointList, center [2]float64, scaling float64, angle float64) ([]float64, []float64) {
	var centerPointList PointList
	centerPointList = append(centerPointList, center)

	//var angleFloatList FloatList
	//angleFloatList = append(angleFloatList, angle*PI/180)

	for i := 0; i < len(X); i++ {
		X[i][0] = X[i][0] - center[0]
		X[i][1] = X[i][1] - center[1]
	}

	//Ndim := X.shape[1]
	Ndim := 2

	var stretch PointList
	var rotTot PointList
	if Ndim == 1 {
		panic("Ndim == 1")
	} else if Ndim == 2 {
		stretch = append(stretch, [2]float64{1, 0})
		stretch = append(stretch, [2]float64{0, scaling})

		rotTot = append(rotTot, [2]float64{math.Cos(-angle), -math.Sin(-angle)})
		rotTot = append(rotTot, [2]float64{math.Sin(-angle), math.Cos(-angle)})
		//rot_tot = np.array([[np.cos(-angle[0]), -np.sin(-angle[0])], [np.sin(-angle[0]), np.cos(-angle[0])]])
	} else if Ndim == 3 {
		panic("Ndim == 3")
	} else {
		panic("Ndim >= 4")
	}

	// TODO np.dot
	//X_adj = np.dot(stretch, np.dot(rotTot, X.T)).T
	X_adj := MatrixTranspose(X)

	for i := 0; i < len(X_adj[0]); i++ {
		X_adj[0][i] = X_adj[0][i] + center[0]
		X_adj[1][i] = X_adj[1][i] + center[1]
	}

	return X_adj[0], X_adj[1]
}

func (krigingPy KrigingPy) _get_kriging_matrix(n int) [][]float64 {
	var d [][]float64
	if krigingPy.CoordinatesType == "euclidean" {
		xy := concatenate(krigingPy.X_ADJUSTED, krigingPy.Y_ADJUSTED)
		d = cdist(xy, xy)
	}

	var a [][]float64
	a = krigingPy.VariogramFunction(krigingPy.VariogramModelParameters, d)
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[i]); j++ {
			a[i][j] = a[i][j] * -1
		}
		a[i] = append(a[i], 0)
	}
	lastLine := make([]float64, n+1)
	a = append(a, lastLine)

	fill_diagonal(a, 0)
	for i := 0; i < len(a[n]); i++ {
		a[n][i] = 1.0
	}

	for i := 0; i < len(a); i++ {
		a[i][n] = 1.0
	}

	a[n][n] = 0.0

	return a
}

func (krigingPy KrigingPy) Execute(style string, xpoints []float64, ypoints []float64, backend string) ([][]float64, [][]float64) {

	n := len(krigingPy.X_ADJUSTED)
	a := krigingPy._get_kriging_matrix(n)
	var npt int
	var xpts []float64
	var ypts []float64

	if style == "grid" {
		npt = len(xpoints) * len(ypoints)
		grid_x, grid_y := Meshgrid(xpoints, ypoints)
		xpts = flatten(grid_x)
		ypts = flatten(grid_y)

	} else {
		panic("style != grid")
	}
	nx := len(xpoints)
	ny := len(ypoints)

	var xy_data PointList
	var xy_points PointList

	if krigingPy.CoordinatesType == "euclidean" {
		var pointList PointList

		for i := 0; i < len(xpts); i++ {
			var point [2]float64
			point[0] = xpts[i]
			point[1] = ypts[i]
			pointList = append(pointList, point)
		}

		xpts, ypts = _adjust_for_anisotropy(pointList, [2]float64{krigingPy.XCENTER, krigingPy.YCENTER},
			krigingPy.AnisotropyScaling, krigingPy.AnisotropyAngle)
		xy_data = concatenate(krigingPy.X_ADJUSTED, krigingPy.Y_ADJUSTED)
		xy_points = concatenate(xpts, ypts)
	} else {
		panic("krigingPy.CoordinatesType != euclidean")
	}

	mask := make([]bool, npt)
	var bd [][]float64
	var zvalues []float64
	var sigmasq []float64
	if krigingPy.CoordinatesType == "euclidean" {
		bd = cdist(xy_points, xy_data)
		zvalues, sigmasq = krigingPy._exec_loop(a, bd, mask)
	} else {
		panic("krigingPy.CoordinatesType != euclidean")
	}

	var zvaluesRes [][]float64
	var sigmasqRes [][]float64
	for i := 0; i < ny; i++ {
		var tmpZvalues []float64
		var tmpSigmasq []float64
		for j := 0; j < nx; j++ {
			tmpZvalues = append(tmpZvalues, zvalues[i*nx+j])
			tmpSigmasq = append(tmpSigmasq, sigmasq[i*nx+j])
		}
		zvaluesRes = append(zvaluesRes, tmpZvalues)
		sigmasqRes = append(sigmasqRes, tmpSigmasq)
	}

	return zvaluesRes, sigmasqRes
}

func (krigingPy KrigingPy) _exec_loop(a [][]float64, bd_all [][]float64, mask []bool) ([]float64, []float64) {
	npt := len(bd_all)
	zvalues := make([]float64, npt)
	sigmasq := make([]float64, npt)

	var a_inv [][]float64
	for i := 0; i < len(a); i++ {
		tmp := make([]float64, len(a))
		a_inv = append(a_inv, tmp)
	}
	a_inv = Gauss(a)
	for i := 0; i < len(bd_all); i++ {
		bd := bd_all[i]
		zero_value := false
		zero_index := 0
		for j := 0; j < len(bd); j++ {
			if bd[j] <= 0 {
				zero_value = true
				zero_index = j
			}
		}
		b := LinearVariogramModelLine(krigingPy.VariogramModelParameters, bd)
		for k := 0; k < len(b); k++ {
			b[k] = b[k] * -1
		}
		b = append(b, 1)
		if zero_value {
			b[zero_index] = 0.0
		}
		x := Dot(a_inv, b)
		for j := 0; j < len(krigingPy.Z); j++ {
			zvalues[i] = zvalues[i] + x[j]*krigingPy.Z[j]
		}
	}

	return zvalues, sigmasq
}
