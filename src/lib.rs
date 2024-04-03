use std::collections::HashSet;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Default, PartialEq)]
pub enum Label {
    Cluster(usize),
    Noise,
    #[default]
    None,
}

pub fn print_labels(labels: &[Label]) {
    for group in labels.chunks(20) {
        for g in group {
            print!("{g:?}");
        }
        println!();
    }
}

impl std::fmt::Debug for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let w = f.width().unwrap_or(4);
        match self {
            Label::Cluster(n) => write!(f, "{n:w$}"),
            Label::Noise => write!(f, "{:>w$}", "-1"),
            Label::None => write!(f, "{:>w$}", "N"),
        }
    }
}

fn range_query(
    db: &impl Fn(usize, usize) -> f64,
    cols: usize,
    p: usize,
    eps: f64,
) -> HashSet<usize> {
    let mut n = HashSet::new();
    for q in 0..cols {
        if db(p, q) <= eps {
            n.insert(q);
        }
    }
    n
}

/// takes a distance matrix and DBSCAN parameters `eps` and `min_pts`. Using the
/// "original query-based algorithm" from Wikipedia for now
pub fn dbscan(
    rows: usize,
    cols: usize,
    db: impl Fn(usize, usize) -> f64,
    eps: f64,
    min_pts: usize,
) -> Vec<Label> {
    let mut label = vec![Label::None; rows];
    let mut c = 0; // cluster counter

    // each row corresponds to one entry's distance from every other molecule
    for p in 0..rows {
        if label[p] != Label::None {
            continue;
        }
        // here we have each of those distances, count up the neighbors of the
        // ith molecule. this is basically range_query
        let n = range_query(&db, cols, p, eps);
        if n.len() < min_pts {
            label[p] = Label::Noise;
            continue;
        }
        label[p] = Label::Cluster(c);
        let mut s = n; // S := N \ {P} since p is counted as its own neighbor
        s.remove(&p);

        let mut s: Vec<_> = s.into_iter().collect();

        let mut qi = 0;
        while qi < s.len() {
            let q = s[qi];
            if label[q] == Label::Noise {
                label[q] = Label::Cluster(c);
            }
            if label[q] != Label::None {
                qi += 1;
                continue;
            }
            label[q] = Label::Cluster(c);
            let neighbors = range_query(&db, cols, q, eps);
            if neighbors.len() >= min_pts {
                for n in neighbors {
                    if !s.contains(&n) {
                        s.push(n);
                    }
                }
            }
            qi += 1;
        }
        c += 1;
    }
    label
}
