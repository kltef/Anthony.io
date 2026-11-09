//! Decision-time benchmark: plays games single-threaded and times every decision of agent A.
//!   usage: bench <agentA> [opponent=t4] [--games N] [--seed S]
//! Prints mean / p95 / max milliseconds per decision (native; the phone's WebAssembly is slower).

use stateio::agent::{play, Agent};
use stateio::rules::*;
use stateio::setup::{new_game, Setup};
use stateio::{roster, Map, Rng};
use std::sync::{Arc, Mutex};
use std::time::Instant;

struct Timed {
    inner: Box<dyn Agent>,
    times: Arc<Mutex<Vec<f64>>>,
}

impl Agent for Timed {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn reset(&mut self, w: &World, me: u8, seed: u64) {
        self.inner.reset(w, me, seed)
    }
    fn act(&mut self, w: &World, s: &State, out: &mut Vec<Action>) {
        let before = self.inner.stats().0;
        let t = Instant::now();
        self.inner.act(w, s, out);
        if self.inner.stats().0 > before {
            self.times.lock().unwrap().push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    fn stats(&self) -> (u64, u64) {
        self.inner.stats()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let (mut games, mut seed) = (4usize, 7u64);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => { games = args[i + 1].parse().unwrap(); i += 1; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 1; }
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    let spec = pos.first().expect("usage: bench <agent> [opponent]").clone();
    let opp = pos.get(1).cloned().unwrap_or("t4".into());
    let map = Map::continental();
    let w = World::new(map.clone(), Rules::standard(&map, 1.0));
    let times = Arc::new(Mutex::new(Vec::new()));
    let mut rollouts = 0u64;
    for g in 0..games {
        let start = new_game(&w.map, &Setup::duel(), &mut Rng::new(seed + g as u64));
        let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(Timed { inner: roster::make(&spec), times: times.clone() }), roster::make(&opp)];
        play(&w, &start, &mut agents, 400.0, seed + g as u64);
        rollouts += agents[0].stats().1;
    }
    let mut t = times.lock().unwrap().clone();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = t.len().max(1);
    let mean = t.iter().sum::<f64>() / n as f64;
    println!(
        "{spec}: {} decisions  mean {:.2} ms  p95 {:.2} ms  max {:.2} ms  rollouts/decision {:.0}",
        t.len(), mean, t[(n * 95 / 100).min(n - 1)], t.last().copied().unwrap_or(0.0), rollouts as f64 / n as f64
    );
}
