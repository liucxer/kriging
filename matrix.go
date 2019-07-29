package kriging

func step0(m [][]float64) [][]float64 {
	n := len(m)

	var l [][]float64
	for i := 0; i < n; i++ {
		tmp := make([]float64, n)
		l = append(l, tmp)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				l[i][j] = 1
			} else {
				l[i][j] = 0
			}
		}
	}

	return l
}

func step1(m [][]float64) ([]int, [][]float64, [][]float64) {
	n := len(m)
	var swap []int
	var l [][]float64

	// 交换操作记录数组 swap
	for i := 0; i < n; i++ {
		swap = append(swap, i)
		l = append(l, make([]float64, n))
	}

	// 对每一列进行操作
	for i := 0; i < n; i++ {
		max_row := m[i][i]
		row := i
		for j := i; j < n; j++ {
			if m[j][i] >= max_row {
				max_row = m[j][i]
				row = j
			}
		}
		swap[i] = row

		// 交换
		if row != i {
			for j := 0; j < n; j++ {
				m[i][j], m[row][j] = m[row][j], m[i][j]
			}
		}

		// 消元
		for j := i + 1; j < n; j++ {
			if m[j][i] != 0 {
				l[j][i] = m[j][i] / m[i][i]
				for k := 0; k < n; k++ {
					m[j][k] = m[j][k] - (l[j][i] * m[i][k])
				}
			}
		}
	}
	return swap, m, l
}

func step2(m [][]float64) ([][]float64, [][]float64) {
	n := len(m)
	long := len(m) - 1

	var l [][]float64

	// 交换操作记录数组 swap
	for i := 0; i < n; i++ {
		l = append(l, make([]float64, n))
	}

	for i := 0; i < n-1; i++ {
		for j := 0; j < long-i; j++ {
			if m[long-i-j-1][long-i] != 0 && m[long-i][long-i] != 0 {
				l[long-i-j-1][long-i] = m[long-i-j-1][long-i] / m[long-i][long-i]
				for k := 0; k < n; k++ {
					m[long-i-j-1][k] = m[long-i-j-1][k] - l[long-i-j-1][long-i]*m[long-i][k]
				}
			}
		}
	}
	return m, l
}

func step3(m [][]float64) []float64 {
	n := len(m)
	var l []float64
	for i := 0; i < n; i++ {
		l = append(l, m[i][i])
	}
	return l
}

func Gauss(matrix [][]float64) [][]float64 {
	n := len(matrix)
	new := step0(matrix)
	swap, matrix1, l1 := step1(matrix)
	matrix2, l2 := step2(matrix1)
	l3 := step3(matrix2)

	for i := 0; i < n; i++ {
		if swap[i] != i {
			new[i], new[swap[i]] = new[swap[i]], new[i]
		}

		for j := i + 1; j < n; j++ {
			for k := 0; k < n; k++ {
				if l1[j][i] != 0 {
					new[j][k] = new[j][k] - l1[j][i]*new[i][k]
				}
			}
		}
	}

	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if l2[n-1-i-j-1][n-1-i] != 0 {
				for k := 0; k < n; k++ {
					new[n-1-i-j-1][k] = new[n-1-i-j-1][k] - l2[n-1-i-j-1][n-i-1]*new[n-1-i][k]
				}
			}
		}
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			new[i][j] = new[i][j] / l3[i]
		}
	}

	return new
}
