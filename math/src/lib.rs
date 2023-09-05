use std::assert_eq;

use rand::{thread_rng, Rng};

#[cfg(test)]
mod tests;

/// Stores a row-major matrix
#[derive(Debug, PartialEq, Clone)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Matrices were not of the right order: ",
        );
        assert_eq!(self.data.len(), self.rows * self.cols, "Malformed matrix");
        assert_eq!(
            other.data.len(),
            other.rows * other.cols,
            "Malformed matrix"
        );

        let mut data = vec![];
        for row_index in 0..self.rows {
            for col_index in 0..other.cols {
                let mut total = 0.0;
                for (a, b) in self.get_row(row_index).zip(other.get_col(col_index)) {
                    total += a * b;
                }
                data.push(total);
            }
        }

        Matrix {
            data,
            rows: self.rows,
            cols: other.cols,
        }
    }
}

macro_rules! matrix_op {
    ($trait:ident, $func_name:ident, $token:tt) => {
        impl std::ops::$trait for &Matrix {
            type Output = Matrix;
            fn $func_name(self, other: Self) -> Self::Output {
                assert_eq!(self.data.len(), self.rows * self.cols, "Malformed matrix");
                assert_eq!(other.data.len(), other.rows * other.cols, "Malformed matrix");
                assert_eq!(self.cols, other.cols, "matrices had different order");
                assert_eq!(self.rows, other.rows, "matrices had different order");

                let mut data = vec![];
                for i in 0..self.data.len() {
                    data.push(self.data[i] $token other.data[i]);
                }

                Matrix {
                    data,
                    rows: self.rows,
                    cols: self.cols,
                }
            }
        }
    };
}

matrix_op!(Add, add, +);
matrix_op!(Sub, sub, -);
matrix_op!(Mul, mul, *);
matrix_op!(Div, div, /);

impl std::ops::AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}

impl Matrix {
    pub fn add_col(&mut self, x: f32) {
        let mut new_data = vec![];
        for row_index in 0..self.rows {
            for col_index in 0..self.cols {
                new_data.push(self.data[row_index * self.cols + col_index]);
            }
            new_data.push(x);
        }
        self.data = new_data;
        self.cols += 1;
    }
    pub fn apply(&self, mut func: impl FnMut(f32) -> f32) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| func(*x)).collect(),
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = vec![];
        for col_index in 0..self.cols {
            for row_index in 0..self.rows {
                result.push(self.data[row_index * self.cols + col_index]);
            }
        }
        Matrix {
            data: result,
            rows: self.cols,
            cols: self.rows,
        }
    }

    pub fn get_row(&self, index: usize) -> impl Iterator<Item = f32> + '_ {
        let start = self.cols * index;
        let end = self.cols * (index + 1);
        self.data[start..end].iter().map(|x| *x)
    }

    pub fn get_col(&self, index: usize) -> impl Iterator<Item = f32> + '_ {
        (0..self.rows).map(move |row_index| self.data[row_index * self.cols + index])
    }

    pub fn from_rows(rows: Vec<Vec<f32>>) -> Matrix {
        Matrix {
            rows: rows.len(),
            cols: rows[0].len(),
            data: rows.into_iter().flatten().collect(),
        }
    }
    pub fn from_dim(rows: usize, cols: usize) -> Matrix {
        let mut rng = thread_rng();
        Matrix {
            rows,
            cols,
            data: (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect(),
        }
    }
    pub fn repr(&self) -> String {
        let mut rows = vec![];
        for row_index in 0..self.rows {
            let mut row = vec![];
            for col_index in 0..self.cols {
                row.push(format!(
                    "{:.5}",
                    self.data[row_index * self.cols + col_index].to_string()
                ));
            }
            if row_index != 0 {
                rows.push(format!("  [ {} ]", row.join(" ")));
            } else {
                rows.push(format!("[ {} ]", row.join(" ")));
            }
        }
        let row_text = rows.join("\n");
        format!("[ {} ]", row_text)
    }
}

impl From<Vec<f32>> for Matrix {
    fn from(value: Vec<f32>) -> Self {
        Matrix::from_rows(vec![value])
    }
}

impl<const T: usize> From<Vec<[f32; T]>> for Matrix {
    fn from(value: Vec<[f32; T]>) -> Self {
        let mut rows = vec![];
        for item in value {
            rows.push(item.to_vec());
        }
        Matrix::from_rows(rows)
    }
}

#[macro_export]
macro_rules! matrix {
    ($rows:expr; $cols:expr) => {{
        Matrix::from_dim($rows, $cols)
    }};
    ( $( $x:expr ),+ ) => {{
        let mut temp = vec![];
        $(
            temp.push($x);
        )*
        Matrix::from(temp)
    }}
}
