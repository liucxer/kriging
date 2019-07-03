package kriging_test

import (
	"github.com/liucxer/kriging"
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"
)

func TestKriging(t *testing.T) {
	var pm = FloatList{46, 36, 40, 43, 52, 123, 85, 154, 172, 216, 226, 216, 204, 198, 204, 46, 181, 59, 63, 180, 196, 55, 70, 59, 61, 61, 80, 70, 63}
	var csvX = FloatList{12965158.453710767, 12958367.964772375, 12936104.066613723, 12957254.769864446, 12964379.217275213, 12928200.3827674, 12929313.577675335, 12985975.198489109, 13037738.761707982, 12986865.754415454, 12958034.006299999, 12969388.59436091, 12946456.779257497, 13000224.093310649, 13025382.298229927, 12911725.098129995, 12966828.246072665, 12909943.986277305, 13014472.988132186, 12945009.625877187, 12956920.811392065, 13058221.54801394, 13128798.105176877, 12985641.240016729, 12834692.010501053, 12756768.366945764, 12916845.794706486, 12947013.376711464, 12877550.014456464}
	var csvY = FloatList{4859405.155303598, 4849389.9128107885, 4866232.918309031, 4863326.913219404, 4856791.510127826, 4828520.948656471, 4840399.303764576, 4884414.7725505745, 4880484.665662185, 4849389.9128107885, 4825047.031816001, 4836196.80391972, 4796433.202106878, 4824178.74151438, 4831996.075600284, 4919125.716495515, 4857081.881192424, 4931991.02785777, 4938722.782062121, 4864053.334723731, 4847939.265060946, 4825191.754209896, 5013964.296751478, 5044106.02465086, 4940040.404148559, 4843588.590391939, 4792393.394419291, 4785615.944314293, 5111376.463640959}

	UseKriging(pm, csvX, csvY, "./example.png")
}

type FloatList []float64

func (t FloatList) min() float64 {
	min := float64(0)
	for i := 0; i < len(t); i++ {
		if min == 0 || min > t[i] {
			min = t[i]
		}
	}

	return min
}

func (t FloatList) max() float64 {
	max := float64(0)
	for i := 0; i < len(t); i++ {
		if max < t[i] {
			max = t[i]
		}
	}

	return max
}

func UseKriging(pm, csvX, csvY FloatList, path string) {
	var model = "exponential"
	var sigma2 = float64(0)
	var alpha = float64(100)
	var kriging kriging.Kriging
	var variogram = kriging.Train(pm, csvX, csvY, model, sigma2, alpha)

	var rangemaxX = csvX.max()
	var rangeminX = csvX.min()
	var rangemaxY = csvY.max()
	var rangeminY = csvY.min()
	var rangemaxPM = pm.max()
	var rangeminPM = pm.min()
	var colorperiod = (rangemaxPM - rangeminPM) / 5
	_ = colorperiod
	var xl = rangemaxX - rangeminX
	var yl = rangemaxY - rangeminY
	var gridX = xl / 100
	var gridY = yl / 100
	var gridPoint = [][2]float64{}
	var krigingValue = []float64{}
	var gX = rangeminX

	for i := 0; i < 100; i++ {
		gX = gX + gridX
		gY := rangeminY
		for j := 0; j < 100; j++ {
			gY = gY + gridY
			var pp = [2]float64{gX, gY}
			krigingValue = append(krigingValue, variogram.Predict(gX, gY))
			gridPoint = append(gridPoint, pp)
		}
	}

	img := image.NewRGBA(image.Rect(0, 0, 100, 100))

	for i := 0; i < 10000; i++ {
		zi := krigingValue[i]
		var color color.RGBA

		if zi <= rangemaxPM && zi > rangemaxPM-colorperiod {
			color.R = 189
			color.G = 0
			color.B = 36
			color.A = 128
		} else if zi <= rangemaxPM-colorperiod && zi > rangemaxPM-2*colorperiod {
			color.R = 240
			color.G = 59
			color.B = 32
			color.A = 128
		} else if zi <= rangemaxPM-2*colorperiod && zi > rangemaxPM-3*colorperiod {
			color.R = 253
			color.G = 141
			color.B = 60
			color.A = 128
		} else if zi <= rangemaxPM-3*colorperiod && zi > rangemaxPM-4*colorperiod {
			color.R = 254
			color.G = 204
			color.B = 92
			color.A = 128
		} else {
			color.R = 255
			color.G = 255
			color.B = 78
			color.A = 128
		}

		x := i % 100
		y := i / 100
		img.Set(x, y, color)
	}
	file, err := os.Create(path)
	if err != nil {
		return
	}
	defer file.Close()
	png.Encode(file, img)
}
