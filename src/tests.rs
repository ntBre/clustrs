use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

use crate::{dbscan, Label};

#[derive(Debug, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Box<[T]>,
}

impl<T> Matrix<T> {
    pub fn new(v: Vec<Vec<T>>) -> Self {
        let rows = v.len();
        let v: Vec<T> = v.into_iter().flatten().collect();
        let cols = v.len() / rows;
        Self {
            rows,
            cols,
            data: v.into_boxed_slice(),
        }
    }

    /// (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.data[self.cols * x + y]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        &mut self.data[self.cols * x + y]
    }
}

impl<T: Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(8);
        let prec = f.precision().unwrap_or(4);
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{col:width$.prec$}", col = &self[(row, col)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
/// the input matrix and output results are taken from this page:
/// scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html, with the
/// distances precomputed using this code:
/// ``` python
/// import numpy as np
/// from sklearn import metrics
/// from sklearn.cluster import DBSCAN
/// from sklearn.datasets import make_blobs
/// from sklearn.preprocessing import StandardScaler
///
/// centers = [[1, 1], [-1, -1], [1, -1]]
/// X, labels_true = make_blobs(
///     n_samples=750, centers=centers, cluster_std=0.4, random_state=0
/// )
///
/// X = StandardScaler().fit_transform(X)
///
/// dist = np.zeros((len(X), len(X)))
/// for i in range(len(X)):
///     for j in range(len(X)):
///         dist[(i, j)] = np.linalg.norm(X[i] - X[j])
///
/// db = DBSCAN(eps=0.3, min_samples=10, metric="precomputed").fit(dist)
/// labels = db.labels_
/// ```
#[test]
fn test_dbscan() {
    let s = std::fs::read_to_string("testfiles/dist.mat").unwrap();
    let mut v: Vec<Vec<f64>> = Vec::new();
    for line in s.lines() {
        v.push(
            line.split_ascii_whitespace()
                .map(|s| s.parse().unwrap())
                .collect(),
        );
    }

    let m = Matrix::new(v);
    let (r, c) = m.shape();
    let got = dbscan(r, c, |i, j| m[(i, j)], 0.3, 10);
    let want: Vec<Label> = std::fs::read_to_string("testfiles/dist.want")
        .unwrap()
        .split_ascii_whitespace()
        .map(|s| {
            let n = s.parse::<isize>().unwrap();
            if n < 0 {
                Label::Noise
            } else {
                Label::Cluster(n as usize)
            }
        })
        .collect();
    assert_eq!(got, want);
}
