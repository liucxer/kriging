package kriging_test

import (
	"git.querycap.com/intelliep/srv-pollution-map-worker/modules/kriging"
	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestDot(t *testing.T) {
	m := [][]float64{{1, 2}, {3, 4}}
	n := []float64{6, 8}

	res := kriging.Dot(m, n)
	spew.Dump(res)
	require.Equal(t, res[0], float64(22))
	require.Equal(t, res[1], float64(50))
}

func TestMeshgrid(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	y := []float64{7, 6, 5}
	resx, resy := kriging.Meshgrid(x, y)
	spew.Dump(resx)
	spew.Dump(resy)
}
