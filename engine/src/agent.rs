//! The agent interface, the scripted agents, and the match runner used by the arena and training.

use crate::policy::{self, PolicyParams};
use crate::rules::*;

pub trait Agent: Send {
    fn name(&self) -> String;
    /// Called once before a game. `me` is the owner id this agent controls.
    fn reset(&mut self, w: &World, me: u8, seed: u64);
    /// Called every tick with the current state. Push any orders for `me` onto `out`.
    fn act(&mut self, w: &World, s: &State, out: &mut Vec<Action>);
}

/// The base policy on a fixed decision cadence.
pub struct Scripted {
    pub label: String,
    pub params: PolicyParams,
    pub cadence: f64,
    me: u8,
    next: f64,
}

impl Scripted {
    pub fn new(label: &str, params: PolicyParams, cadence: f64) -> Scripted {
        Scripted { label: label.to_string(), params, cadence, me: 0, next: 0.0 }
    }
}

impl Agent for Scripted {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn reset(&mut self, _w: &World, me: u8, seed: u64) {
        self.me = me;
        // stagger the first decision a little so mirrored bots are not perfectly in phase
        self.next = 0.2 + (seed % 7) as f64 * 0.03;
    }
    fn act(&mut self, w: &World, s: &State, out: &mut Vec<Action>) {
        if s.time < self.next {
            return;
        }
        self.next = s.time + self.cadence;
        policy::act(w, s, self.me, &self.params, out);
    }
}

/// Does nothing (sanity baseline).
pub struct Idle;
impl Agent for Idle {
    fn name(&self) -> String {
        "idle".into()
    }
    fn reset(&mut self, _w: &World, _me: u8, _seed: u64) {}
    fn act(&mut self, _w: &World, _s: &State, _out: &mut Vec<Action>) {}
}

#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// Winning owner, or None for a draw / timeout without a clear leader.
    pub winner: Option<u8>,
    pub time: f64,
    pub timed_out: bool,
    /// Orders issued per owner (index = owner id).
    pub orders: Vec<u32>,
    /// Orders the rules rejected, per owner.
    pub rejected: Vec<u32>,
}

/// Rough material strength: troops plus a growth term (used only to adjudicate timeouts).
pub fn strength(w: &World, s: &State, owner: u8) -> f64 {
    let (troops, _) = s.totals(owner);
    let mut g = 0.0;
    for i in 0..s.n {
        if s.owner[i] == owner {
            g += s.growth(w, i);
        }
    }
    troops + 45.0 * g
}

pub const TICK: f64 = 0.05;

/// Play one game. `agents[k]` controls owner `k + 1`.
pub fn play(w: &World, start: &State, agents: &mut [Box<dyn Agent>], max_time: f64, seed: u64) -> Outcome {
    let mut s = start.clone();
    let teams = agents.len();
    for (k, a) in agents.iter_mut().enumerate() {
        a.reset(w, (k + 1) as u8, seed.wrapping_add(k as u64 * 977));
    }
    let mut out = Outcome { orders: vec![0; teams + 1], rejected: vec![0; teams + 1], ..Default::default() };
    let mut buf: Vec<Vec<Action>> = vec![Vec::new(); teams];
    let mut tick = 0u64;
    s.recompute_supply(w);
    loop {
        let alive = s.alive();
        // every agent decides from the same position, then orders apply in rotating order
        for k in 0..teams {
            buf[k].clear();
            if alive >> (k + 1) & 1 == 1 {
                agents[k].act(w, &s, &mut buf[k]);
            }
        }
        for r in 0..teams {
            let k = (r + tick as usize) % teams;
            for &a in &buf[k] {
                out.orders[k + 1] += 1;
                if !s.apply(w, (k + 1) as u8, a) {
                    out.rejected[k + 1] += 1;
                }
            }
        }
        s.step(w, TICK);
        tick += 1;
        if let Some(win) = s.winner() {
            out.winner = Some(win);
            break;
        }
        if s.alive() == 0 {
            break;
        }
        if s.time >= max_time {
            out.timed_out = true;
            let mut best = (0u8, -1.0f64);
            let mut second = -1.0f64;
            for o in 1..=teams as u8 {
                let v = if s.alive() >> o & 1 == 1 { strength(w, &s, o) } else { 0.0 };
                if v > best.1 {
                    second = best.1;
                    best = (o, v);
                } else if v > second {
                    second = v;
                }
            }
            if best.1 > 1.5 * second.max(1.0) {
                out.winner = Some(best.0);
            }
            break;
        }
    }
    out.time = s.time;
    out
}
