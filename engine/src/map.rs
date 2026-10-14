//! Static board description: territory positions and the border graph.

pub const MAXN: usize = 64;

#[derive(Clone, Debug)]
pub struct Map {
    pub name: String,
    pub n: usize,
    pub names: Vec<String>,
    pub cx: Vec<f64>,
    pub cy: Vec<f64>,
    pub adj: Vec<Vec<u8>>,
    /// Bitmask form of `adj` (bit j set = borders j).
    pub adj_mask: Vec<u64>,
    /// Euclidean centroid distance, row-major n*n. `hypot || 1` like the game.
    pub dist: Vec<f64>,
    /// Diagonal of the centroid bounding box (the game's `rlDiag`).
    pub diag: f64,
}

impl Map {
    pub fn new(name: &str, names: Vec<String>, cx: Vec<f64>, cy: Vec<f64>, adj: Vec<Vec<u8>>) -> Map {
        let n = cx.len();
        assert!(n <= MAXN && cy.len() == n && adj.len() == n);
        let mut dist = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = (cx[j] - cx[i]).hypot(cy[j] - cy[i]);
                dist[i * n + j] = if d == 0.0 { 1.0 } else { d };
            }
        }
        let (mut minx, mut miny, mut maxx, mut maxy) = (1e9f64, 1e9f64, -1e9f64, -1e9f64);
        for i in 0..n {
            minx = minx.min(cx[i]);
            miny = miny.min(cy[i]);
            maxx = maxx.max(cx[i]);
            maxy = maxy.max(cy[i]);
        }
        let diag = (maxx - minx).hypot(maxy - miny);
        let adj_mask = adj.iter().map(|a| a.iter().fold(0u64, |m, &j| m | (1u64 << j))).collect();
        Map { name: name.to_string(), n, names, cx, cy, adj, adj_mask, dist, diag: if diag == 0.0 { 1.0 } else { diag } }
    }

    #[inline]
    pub fn d(&self, a: usize, b: usize) -> f64 {
        self.dist[a * self.n + b]
    }

    #[inline]
    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        self.adj_mask[a] >> b & 1 == 1
    }

    /// Parse the `maps.txt` format written by `tools/export_maps.js`.
    pub fn parse_all(text: &str) -> Vec<Map> {
        let mut maps = Vec::new();
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        while let Some(h) = lines.next() {
            let p: Vec<&str> = h.split('|').collect();
            assert_eq!(p[0], "map", "bad map header: {h}");
            let n: usize = p[2].parse().unwrap();
            let (mut names, mut cx, mut cy, mut adj) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for _ in 0..n {
                let l = lines.next().expect("truncated map");
                let q: Vec<&str> = l.split('|').collect();
                names.push(q[0].to_string());
                cx.push(q[1].parse().unwrap());
                cy.push(q[2].parse().unwrap());
                adj.push(q[3].split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect());
            }
            maps.push(Map::new(p[1], names, cx, cy, adj));
        }
        maps
    }

    /// The maps shipped with the game (East Coast, Midwest, West Coast, Continental U.S.).
    pub fn builtin() -> Vec<Map> {
        Map::parse_all(include_str!("../data/maps.txt"))
    }

    /// The default solo map.
    pub fn continental() -> Map {
        Map::builtin().into_iter().find(|m| m.name == "Continental U.S.").unwrap()
    }
}
