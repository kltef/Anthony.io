//! The search agent: every decision, enumerate candidate orders, play each one forward in the real
//! rules against models of the opponent, and commit to the orders that beat waiting.
//!
//! This is policy improvement by rollout. There are no scripted behaviours here: hunting,
//! waiting, massing, investing and defending all fall out of "which future scores best".

use crate::agent::Agent;
use crate::policy::{self, PolicyParams, View};
use crate::rules::*;

#[derive(Clone, Debug)]
pub struct SearchParams {
    /// Seconds between decisions.
    pub cadence: f64,
    /// How far each rollout looks ahead, in game seconds.
    pub horizon: f64,
    /// Simulation step inside rollouts.
    pub dt: f64,
    /// How often the simulated players re-decide inside a rollout.
    pub policy_period: f64,
    /// Most orders committed in one decision (salvo width).
    pub max_orders: usize,
    /// Most candidates evaluated per commit round (0 = all).
    pub max_candidates: usize,
    /// A candidate must beat waiting by this much (value units, range -1..1) to be played.
    pub min_gain: f64,
    /// Seconds of growth a state is worth in the leaf evaluation.
    pub kg: f64,
    /// How my own later moves are played inside rollouts.
    pub self_model: PolicyParams,
    /// Whole-position plans tried first: each script's full set of orders is one candidate.
    pub portfolio: Vec<PolicyParams>,
    /// Rounds of single-order refinement after the portfolio step.
    pub refine_rounds: usize,
    /// Opponent models. Each candidate is rolled out against every one.
    pub opp_models: Vec<PolicyParams>,
    /// 0 = average over opponent models, 1 = worst case. In between blends the two.
    pub pessimism: f64,
}

/// The league-tuned base policy (see `bin/tune.rs`).
fn tuned() -> PolicyParams {
    crate::roster::policy("t4").unwrap_or_default()
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            cadence: 0.25,
            horizon: 16.0,
            dt: 0.25,
            policy_period: 0.75,
            max_orders: 4,
            max_candidates: 0,
            min_gain: 0.004,
            kg: 45.0,
            portfolio: vec![tuned(), PolicyParams::rush(), PolicyParams::default()],
            refine_rounds: 2,
            self_model: tuned(),
            opp_models: vec![tuned(), PolicyParams::rush()],
            pessimism: 0.5,
        }
    }
}

impl SearchParams {
    /// Set one option from text. Policy-valued options take names from the roster, `+`-separated.
    pub fn set(&mut self, key: &str, val: &str) {
        let num = || val.parse::<f64>().unwrap_or_else(|_| panic!("bad number for {key}: {val}"));
        let pol = |n: &str| crate::roster::policy(n).unwrap_or_else(|| panic!("unknown policy {n}"));
        match key {
            "cadence" => self.cadence = num(),
            "h" | "horizon" => self.horizon = num(),
            "dt" => self.dt = num(),
            "period" => self.policy_period = num(),
            "orders" => self.max_orders = num() as usize,
            "cands" => self.max_candidates = num() as usize,
            "gain" => self.min_gain = num(),
            "kg" => self.kg = num(),
            "r" | "refine" => self.refine_rounds = num() as usize,
            "pess" => self.pessimism = num(),
            "self" => self.self_model = pol(val),
            "opp" => self.opp_models = val.split('+').map(pol).collect(),
            "port" => self.portfolio = val.split('+').map(pol).collect(),
            _ => panic!("unknown search option {key}"),
        }
    }
}

/// Why an order was (or was not) played. One per decision; kept for the decision log.
#[derive(Clone, Debug, Default)]
pub struct Decision {
    pub time: f64,
    /// Value of doing nothing right now.
    pub wait_value: f64,
    /// Orders committed, each with the value it was chosen at.
    pub chosen: Vec<(Action, f64)>,
    pub candidates: usize,
    pub rollouts: usize,
}

pub struct Searcher {
    pub label: String,
    pub p: SearchParams,
    me: u8,
    next: f64,
    scratch: State,
    work: State,
    acts: Vec<Action>,
    cands: Vec<Action>,
    pub last: Decision,
    pub total_rollouts: u64,
    pub decisions: u64,
}

/// Leaf evaluation in [-1, 1] from `me`'s point of view: material plus the value of future growth.
pub fn evaluate(w: &World, s: &State, me: u8, kg: f64) -> f64 {
    let mut st = [0.0f64; MAX_OWNERS];
    let mut seen = 0u128;
    for i in 0..s.n {
        let o = s.owner[i];
        if o != NEUTRAL {
            st[o as usize] += s.troops[i] + kg * s.growth(w, i);
            seen |= 1u128 << o;
        }
    }
    for o in &s.orbs {
        st[o.owner as usize] += o.count;
        seen |= 1u128 << o.owner;
    }
    if seen >> me & 1 == 0 {
        return -1.0;
    }
    let mine = st[me as usize];
    let (mut top, mut sum) = (0.0f64, 0.0f64);
    let mut m = seen & !(1u128 << me);
    if m == 0 {
        return 1.0;
    }
    while m != 0 {
        let o = m.trailing_zeros() as usize;
        m &= m - 1;
        top = top.max(st[o]);
        sum += st[o];
    }
    let opp = top + 0.5 * (sum - top);
    (mine - opp) / (mine + opp + 1.0)
}

impl Searcher {
    pub fn new(label: &str, p: SearchParams) -> Searcher {
        Searcher {
            label: label.to_string(),
            p,
            me: 0,
            next: 0.0,
            scratch: State::empty(0),
            work: State::empty(0),
            acts: Vec::with_capacity(64),
            cands: Vec::with_capacity(512),
            last: Decision::default(),
            total_rollouts: 0,
            decisions: 0,
        }
    }

    /// Which owner this searcher plays (set by `Agent::reset` in the arena).
    pub fn set_owner(&mut self, me: u8) {
        self.me = me;
    }

    /// Play `self.scratch` forward for the horizon against one opponent model; return its value.
    fn rollout(&mut self, w: &World, opp: &PolicyParams) -> f64 {
        let me = self.me;
        let p = &self.p;
        let st = &mut self.scratch;
        let end = st.time + p.horizon;
        let others = st.alive() & !(1u128 << me);
        let mut next_opp = st.time + 0.1;
        let mut next_me = st.time + p.policy_period;
        let mut k = 0u32;
        while st.time < end {
            if st.time >= next_opp {
                next_opp += p.policy_period;
                let mut m = others;
                while m != 0 {
                    let o = m.trailing_zeros() as u8;
                    m &= m - 1;
                    self.acts.clear();
                    policy::act(w, st, o, opp, &mut self.acts);
                    for &a in &self.acts {
                        st.apply(w, o, a);
                    }
                }
            }
            if st.time >= next_me {
                next_me += p.policy_period;
                self.acts.clear();
                policy::act(w, st, me, &p.self_model, &mut self.acts);
                for &a in &self.acts {
                    st.apply(w, me, a);
                }
            }
            st.step(w, p.dt);
            k += 1;
            if k % 4 == 0 {
                let alive = st.alive();
                // decided inside the horizon: sooner wins and later losses score better, so a
                // winning position still has a best move instead of a tie between all of them
                if alive >> me & 1 == 0 {
                    return -1.0 - 0.5 * (end - st.time) / p.horizon;
                }
                if alive & others == 0 {
                    return 1.0 + 0.5 * (end - st.time) / p.horizon;
                }
            }
        }
        evaluate(w, st, me, p.kg)
    }

    /// Value of position `base` after optionally applying `a`, across the opponent models.
    fn value_of(&mut self, w: &World, base: &State, acts: &[Action]) -> Option<f64> {
        let (mut sum, mut worst) = (0.0, 1e9f64);
        let nm = self.p.opp_models.len();
        for mi in 0..nm {
            self.scratch.copy_from(base);
            let mut any = acts.is_empty();
            for &a in acts {
                any |= self.scratch.apply(w, self.me, a);
            }
            if !any {
                return None;
            }
            let opp = self.p.opp_models[mi].clone();
            let v = self.rollout(w, &opp);
            self.total_rollouts += 1;
            self.last.rollouts += 1;
            sum += v;
            worst = worst.min(v);
        }
        let mean = sum / nm as f64;
        Some(mean + self.p.pessimism * (worst - mean))
    }

    /// Every order worth considering from `s`.
    fn candidates(&mut self, w: &World, s: &State, used: u64) {
        self.cands.clear();
        let me = self.me;
        let v = View::new(w, s, me);
        let mut m = v.mine & !used;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            if s.can_invest(w, me, i) {
                self.cands.push(Action::Invest(i as u8));
            }
            if s.can_upgrade(w, me, i) && s.troops[i] >= s.cap(i) - 10.0 {
                self.cands.push(Action::Upgrade(i as u8));
            }
            let have = s.troops[i].floor();
            if have < 2.0 {
                continue;
            }
            // attacks on every neighbour I do not own: everything, and the smallest force that takes it
            let mut t = w.map.adj_mask[i] & !v.mine;
            while t != 0 {
                let j = t.trailing_zeros() as usize;
                t &= t - 1;
                self.cands.push(Action::Send { src: i as u8, dst: j as u8, amount: have });
                let (po, pt, _) = policy::at_arrival(w, s, &v, i, j);
                if po != me {
                    let need = (need_to_take(pt, s.tile[j], s.def_mult(w, j)) * 1.2 + 2.0).ceil();
                    if need < have * 0.8 {
                        self.cands.push(Action::Send { src: i as u8, dst: j as u8, amount: need });
                    }
                }
            }
            // reinforcement: to the nearest few border states in my territory (and any state under attack)
            if have >= 4.0 {
                let comp = State::flood(w, v.mine, i) & !(1u64 << i);
                let mut dest = comp & (v.border | v.hostile_inbound);
                let mut picked = 0;
                while dest != 0 && picked < 3 {
                    // nearest remaining
                    let (mut best, mut bd) = (usize::MAX, 1e18);
                    let mut d = dest;
                    while d != 0 {
                        let k = d.trailing_zeros() as usize;
                        d &= d - 1;
                        if w.map.d(i, k) < bd {
                            bd = w.map.d(i, k);
                            best = k;
                        }
                    }
                    dest &= !(1u64 << best);
                    picked += 1;
                    self.cands.push(Action::Send { src: i as u8, dst: best as u8, amount: have });
                }
            }
        }
    }

    /// Choose this instant's orders for `me` from position `s`.
    pub fn decide(&mut self, w: &World, s: &State, out: &mut Vec<Action>) {
        self.last = Decision { time: s.time, ..Default::default() };
        self.decisions += 1;
        self.work.copy_from(s);
        let mut used = 0u64;
        let base = self.work.clone();
        let mut cur = match self.value_of(w, &base, &[]) {
            Some(v) => v,
            None => return,
        };
        self.last.wait_value = cur;

        // 1. Portfolio: which whole plan is best from here?
        let mut plan: Vec<Action> = Vec::new();
        let mut best_plan: Vec<Action> = Vec::new();
        let mut best_v = cur + self.p.min_gain;
        for k in 0..self.p.portfolio.len() {
            plan.clear();
            let pp = self.p.portfolio[k].clone();
            policy::act(w, &base, self.me, &pp, &mut plan);
            if plan.is_empty() {
                continue;
            }
            self.last.candidates += 1;
            if let Some(v) = self.value_of(w, &base, &plan) {
                if v > best_v {
                    best_v = v;
                    best_plan.clear();
                    best_plan.extend_from_slice(&plan);
                }
            }
        }
        if !best_plan.is_empty() {
            cur = best_v;
            for &a in &best_plan {
                if self.work.apply(w, self.me, a) {
                    out.push(a);
                    self.last.chosen.push((a, best_v));
                    used |= 1u64 << src_of(a);
                }
            }
        }

        // 2. Refinement: single orders on top of the plan, kept only if they beat it.
        for _round in 0..self.p.refine_rounds {
            let base = self.work.clone();
            self.candidates(w, &base, used);
            let cands = std::mem::take(&mut self.cands);
            self.last.candidates += cands.len();
            let limit = if self.p.max_candidates == 0 { cands.len() } else { self.p.max_candidates.min(cands.len()) };
            let (mut best, mut best_v) = (None, cur + self.p.min_gain);
            for &a in cands.iter().take(limit) {
                if let Some(v) = self.value_of(w, &base, &[a]) {
                    if v > best_v {
                        best_v = v;
                        best = Some(a);
                    }
                }
            }
            self.cands = cands;
            match best {
                Some(a) => {
                    self.work.apply(w, self.me, a);
                    out.push(a);
                    self.last.chosen.push((a, best_v));
                    cur = best_v;
                    used |= 1u64 << src_of(a);
                }
                None => break,
            }
        }
    }
}

#[inline]
fn src_of(a: Action) -> u8 {
    match a {
        Action::Send { src, .. } => src,
        Action::Invest(i) | Action::Upgrade(i) => i,
    }
}

impl Agent for Searcher {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn reset(&mut self, _w: &World, me: u8, seed: u64) {
        self.me = me;
        self.next = 0.1 + (seed % 5) as f64 * 0.02;
    }
    fn act(&mut self, w: &World, s: &State, out: &mut Vec<Action>) {
        if s.time < self.next {
            return;
        }
        self.next = s.time + self.p.cadence;
        self.decide(w, s, out);
    }
}
