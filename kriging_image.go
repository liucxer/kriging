package kriging

import (
	"git.querycap.com/intelliep/srv-pollution-map-worker/modules/color_evaluate"
	"github.com/go-courier/geography"
	"image"
	"image/color"
)

// xResolution:单个格子的宽度
// yResolution:单个格子的高度
func GenerateImage(v []float64, x []float64, y []float64, xResolution float64, yResolution float64, level []float64, minX, maxX, minY, maxY float64) (*image.RGBA, error) {
	value := NewFloatList(v)

	xTrans := make([]float64, len(x))
	yTrans := make([]float64, len(x))
	// 坐标转换
	for i := 0; i < len(x); i++ {
		tmpX, tmpY := geography.WGS84ToPseudoMercator(y[i], x[i])
		xTrans[i] = tmpX
		yTrans[i] = tmpY
	}

	minX, minY = geography.WGS84ToPseudoMercator(minY, minX)
	maxX, maxY = geography.WGS84ToPseudoMercator(maxY, maxX)

	csvX := NewFloatList(xTrans)
	csvY := NewFloatList(yTrans)

	variogramParameters := []float64{10000.0, 0.0001}
	krigingPy := NewKrigingPy(csvX.V, csvY.V, value.V, "linear", variogramParameters, nil, 6, false, 1.0, 0.0, false, false, false, "euclidean")

	var gridXNum = int((maxX - minX) / xResolution)
	var gridYNum = int((maxY - minY) / yResolution)

	var grid_x []float64
	var grid_y []float64
	for i := 0; i < gridXNum; i++ {
		grid_x = append(grid_x, minX+float64(i)*xResolution)
	}

	for i := 0; i < gridYNum; i++ {
		grid_y = append(grid_y, minY+float64(i)*yResolution)
	}

	z, _ := krigingPy.Execute("grid", grid_x, grid_y, "loop")

	return GenerateImg(z, gridXNum, gridYNum, level)
}

var GreenColor = color.RGBA{R: 0, G: 128, B: 0, A: 255}
var YellowColor = color.RGBA{R: 255, G: 255, B: 0, A: 255}
var OrangeColor = color.RGBA{R: 255, G: 126, B: 0, A: 255}
var RedColor = color.RGBA{R: 255, G: 0, B: 0, A: 255}
var PurpleColor = color.RGBA{R: 153, G: 0, B: 76, A: 255}
var BrownColor = color.RGBA{R: 126, G: 0, B: 35, A: 255}

// 黄色加蓝色等于绿色
func GenerateImg(krigingValue [][]float64, gridXNum, gridYNum int, level []float64) (*image.RGBA, error) {
	img := image.NewRGBA(image.Rect(0, 0, gridXNum, gridYNum))
	for i := 0; i < gridYNum; i++ {
		for j := 0; j < gridXNum; j++ {
			var color color.RGBA
			zi := krigingValue[i][j]
			// 绿、黄、橙、红、紫、褐
			if zi < level[0] {
				color = GreenColor
			} else if zi <= level[1] && zi > level[0] {
				color = color_evaluate.Evaluate((zi-level[0])/(level[1]-level[0]), GreenColor, YellowColor)
			} else if zi <= level[2] && zi > level[1] {
				color = OrangeColor
				color = color_evaluate.Evaluate((zi-level[1])/(level[2]-level[1]), YellowColor, OrangeColor)
			} else if zi <= level[3] && zi > level[2] {
				color = color_evaluate.Evaluate((zi-level[2])/(level[3]-level[2]), OrangeColor, RedColor)
			} else if zi <= level[4] && zi > level[3] {
				color = color_evaluate.Evaluate((zi-level[3])/(level[4]-level[3]), RedColor, PurpleColor)
			} else {
				color = BrownColor
			}
			color.A = 255
			img.Set(j, i, color)
		}
	}

	return img, nil
}
