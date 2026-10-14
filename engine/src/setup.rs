//! Board setup: the same procedure as `newGame()` in the original game.

use crate::map::Map;
use crate::rng::Rng;
use crate::rules::*;

#[derive(Clone, Debug)]
pub struct Setup {
    pub neutral_min: f64,
    pub neutral_max: f64,
    /// Starting troops per team, in owner order (index 0 = owner 1).
    pub start: Vec<f64>,
    /// Sprinkle special tiles on ~12% of the neutral states.
    pub tiles: bool,
}

impl Setup {
    /// A symmetric duel with the Nightmare-tier numbers (22 vs 22, neutrals 18..55).
    pub fn duel() -> Setup {
        Setup { neutral_min: 18.0, neutral_max: 55.0, start: vec![22.0, 22.0], tiles: true }
    }
}

pub fn new_game(map: &Map, setup: &Setup, rng: &mut Rng) -> State {
    let n = map.n;
    let mut s = State::empty(n);
    for i in 0..n {
        s.troops[i] = rng.range(setup.neutral_min, setup.neutral_max).floor();
    }
    let teams = setup.start.len().min(n);
    let mut starters: Vec<usize> = Vec::new();
    for t in 0..teams {
        let pick = if t == 0 {
            rng.below(n)
        } else {
            // farthest from every existing starter (max of min squared distance)
            let (mut best, mut best_d) = (0, -1.0);
            for i in 0..n {
                if starters.contains(&i) {
                    continue;
                }
                let mut d = 1e18f64;
                for &st in &starters {
                    let (dx, dy) = (map.cx[st] - map.cx[i], map.cy[st] - map.cy[i]);
                    d = d.min(dx * dx + dy * dy);
                }
                if d > best_d {
                    best_d = d;
                    best = i;
                }
            }
            best
        };
        s.owner[pick] = (t + 1) as u8;
        s.troops[pick] = setup.start[t];
        starters.push(pick);
    }
    if setup.tiles {
        let mut pool: Vec<usize> = (0..n).filter(|i| s.owner[*i] == NEUTRAL).collect();
        rng.shuffle(&mut pool);
        let want = ((n as f64 * 0.12).round() as usize).max(2).min(pool.len());
        for &i in pool.iter().take(want) {
            let r = rng.f64();
            s.tile[i] = if r < 0.40 { TILE_FACTORY } else if r < 0.75 { TILE_FORTRESS } else { TILE_CAPITAL };
        }
    }
    s
}
