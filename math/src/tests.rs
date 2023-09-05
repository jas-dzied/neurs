use crate::{matrix, Matrix};

#[test]
fn constructor() {
    let m1 = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let m2 = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        rows: 2,
        cols: 3,
    };
    assert_eq!(m1, m2)
}

#[test]
fn mat_mul_1() {
    let m1 = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let m2 = Matrix::from_rows(vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]]);
    assert_eq!(
        m1.multiply(&m2),
        Matrix::from_rows(vec![vec![30.0, 36.0], vec![66.0, 81.0],])
    )
}

#[test]
fn mat_mul_2() {
    let m1 = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let m2 = Matrix::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    assert_eq!(
        m1.multiply(&m2),
        Matrix::from_rows(vec![vec![19.0, 22.0], vec![43.0, 50.0],])
    )
}

#[test]
fn mat_mul_3() {
    let m1 = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let m2 = Matrix::from_rows(vec![vec![1.0], vec![2.0], vec![3.0]]);
    assert_eq!(
        m1.multiply(&m2),
        Matrix::from_rows(vec![vec![14.0], vec![32.0],])
    )
}

#[test]
fn mat_transpose() {
    let m1 = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let m2 = Matrix::from_rows(vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    assert_eq!(m1.transpose(), m2)
}

#[test]
fn add_col() {
    let mut m1 = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    m1.add_col(0.0);
    let m2 = Matrix::from_rows(vec![vec![1.0, 2.0, 0.0], vec![3.0, 4.0, 0.0]]);
    assert_eq!(m1, m2)
}

#[test]
fn add() {
    let m1 = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let m2 = Matrix::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let m3 = Matrix::from_rows(vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    assert_eq!(&m1 + &m2, m3)
}

#[test]
fn macro_1() {
    let m1 = matrix!(5; 3).apply(|_| 0.0);
    let m2 = Matrix::from_dim(5, 3).apply(|_| 0.0);
    assert_eq!(m1, m2)
}

#[test]
fn macro_2() {
    let m1 = matrix![1.0, 2.0, 3.0];
    let m2 = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0]]);
    assert_eq!(m1, m2)
}

#[test]
fn macro_3() {
    let m1 = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let m2 = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    assert_eq!(m1, m2)
}
