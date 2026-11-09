//! The planner: searches over multi-step plans instead of single orders, tests every plan
//! against many sampled opponents, and commits to the plan it picks.
//!
//! A plan is a list of timed orders. The important kind is a *strike*: feeders send their troops
//! into a staging state now, the stage waits (for the feeders to land and for growth), then the
//! whole stack hits the target as one orb. Under Lanchester combat one big landing is worth far
//! more than the same troops landing one after another, so this is how a strong player wins, and
//! a planner that only chooses "what to do this instant" can never find it: the first step
//! (moving troops inside its own land) looks pointless unless the strike that follows is part of
//! the same simulated future.
//!
//! Every candidate plan is played forward in the real rules. Inside the rollout the plan's orders
//! fire on schedule, the states the plan has reserved are left alone, and everything else is
//! played by the scripted policy. Opponents are sampled from a pool of styles with randomised
//! parameters (the same samples for every candidate, so the comparison is fair). The score
//! weights the near future more than the far future and blends the average outcome with the
//! worst one.
//!
//! Commitment: the chosen plan is kept between decisions. Each decision re-scores what is left of
//! it next to fresh alternatives, with a small bonus for continuing, and drops it outright when
//! its premises break (a reserved state is lost, the target is already taken). A small move by the
//! opponent does not reset the plan; a move that makes it bad does.

use crate::agent::Agent;
use crate::policy::{self, PolicyParams, View};
use crate::rng::Rng;
use crate::rules::*;
use crate::search::evaluate;

/// "Send everything" (the rules floor the amount to what the state holds).
pub const ALL: f64 = 1e9;

/// Weights of the four checkpoints (25/50/75/100% of the horizon): the near future counts, the
/// far future counts most because it shows the payoff, but no single far guess decides alone.
const CHECK_AT: [f64; 4] = [0.25, 0.5, 0.75, 1.0];
const CHECK_W: [f64; 4] = [0.1, 0.2, 0.3, 0.4];

/// Seconds during which the exact reverse of a recent own-to-own move is not considered.
const REINF_COOLDOWN: f64 = 6.0;

#[derive(Clone, Copy, Debug)]
pub struct Op {
    /// Absolute game time at which the order fires.
    pub at: f64,
    pub act: Action,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Wait,
    Script,
    Direct,
    Strike,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub kind: Kind,
    /// Orders in firing order.
    pub ops: Vec<Op>,
    pub target: Option<u8>,
    pub stage: Option<u8>,
}

impl Plan {
    fn wait() -> Plan {
        Plan { kind: Kind::Wait, ops: Vec::new(), target: None, stage: None }
    }
    fn src(a: Action) -> u8 {
        match a {
            Action::Send { src, .. } => src,
            Action::Invest(i) | Action::Upgrade(i) => i,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlanParams {
    /// Seconds between decisions.
    pub cadence: f64,
    /// Rollout length in game seconds.
    pub horizon: f64,
    pub dt: f64,
    /// How often simulated players re-decide inside a rollout.
    pub policy_period: f64,
    /// Opponent samples used to screen every candidate.
    pub screen: usize,
    /// Opponent samples used for the final comparison of the best candidates.
    pub samples: usize,
    /// How many screened candidates get the full comparison.
    pub final_k: usize,
    /// Most candidates screened per decision (0 = all).
    pub max_cands: usize,
    /// Extra seconds a strike may wait at its stage (on top of the feeders' travel).
    pub waits: Vec<f64>,
    /// Most feeder states per strike.
    pub max_feeders: usize,
    /// Longest feeder trip considered, in seconds.
    pub feed_eta: f64,
    /// Bonus for continuing the current plan (value units, range about -1..1).
    pub commit_bonus: f64,
    /// A new plan must beat waiting by this much.
    pub min_gain: f64,
    /// 0 = average over opponent samples, 1 = worst case.
    pub pessimism: f64,
    /// Relative random perturbation of sampled opponents' parameters.
    pub noise: f64,
    /// Weaker tiers: pick at random among final candidates within this much of the best.
    pub temperature: f64,
    /// Rounds of single extra orders added on top of the chosen plan.
    pub refine: usize,
    /// Seconds of growth a state is worth in the leaf evaluation.
    pub kg: f64,
    /// How my own unplanned moves are played inside rollouts.
    pub self_model: PolicyParams,
    /// Whole-position scripted plans offered as candidates.
    pub portfolio: Vec<PolicyParams>,
    /// Opponent styles; sample j uses style j mod len, randomised by `noise`.
    pub pool: Vec<PolicyParams>,
}

fn tuned() -> PolicyParams {
    crate::roster::policy("t4").unwrap_or_default()
}

impl Default for PlanParams {
    fn default() -> Self {
        PlanParams {
            cadence: 0.3,
            horizon: 14.0,
            dt: 0.25,
            policy_period: 0.75,
            screen: 2,
            samples: 6,
            final_k: 6,
            max_cands: 0,
            waits: vec![0.0, 3.0, 7.0],
            max_feeders: 4,
            feed_eta: 5.0,
            commit_bonus: 0.04,
            min_gain: 0.003,
            pessimism: 0.5,
            noise: 0.25,
            temperature: 0.0,
            refine: 1,
            kg: 45.0,
            self_model: tuned(),
            portfolio: vec![tuned(), PolicyParams::rush(), PolicyParams::default()],
            pool: vec![tuned(), PolicyParams::rush(), PolicyParams::farmer(), PolicyParams::default(), PolicyParams::economist()],
        }
    }
}

impl PlanParams {
    /// Set one option from text. Policy lists take roster names separated by `+`; `waits` takes
    /// seconds separated by `/`.
    pub fn set(&mut self, key: &str, val: &str) -> Result<(), String> {
        let num = || val.parse::<f64>().map_err(|_| format!("bad number for {key}: {val}"));
        let pols = || -> Result<Vec<PolicyParams>, String> {
            val.split('+').map(|n| crate::roster::policy(n).ok_or(format!("unknown policy {n}"))).collect()
        };
        match key {
            "cadence" => self.cadence = num()?,
            "h" | "horizon" => self.horizon = num()?,
            "dt" => self.dt = num()?,
            "period" => self.policy_period = num()?,
            "screen" => self.screen = num()?.max(1.0) as usize,
            "samples" => self.samples = num()?.max(1.0) as usize,
            "k" => self.final_k = num()?.max(1.0) as usize,
            "cands" => self.max_cands = num()? as usize,
            "waits" => {
                self.waits = val.split('/').map(|x| x.parse::<f64>().map_err(|_| format!("bad wait {x}"))).collect::<Result<_, _>>()?
            }
            "feeders" => self.max_feeders = num()? as usize,
            "feed_eta" => self.feed_eta = num()?,
            "bonus" => self.commit_bonus = num()?,
            "gain" => self.min_gain = num()?,
            "pess" => self.pessimism = num()?,
            "noise" => self.noise = num()?,
            "temp" => self.temperature = num()?,
            "refine" => self.refine = num()? as usize,
            "kg" => self.kg = num()?,
            "self" => self.self_model = crate::roster::policy(val).ok_or(format!("unknown policy {val}"))?,
            "port" => self.portfolio = pols()?,
            "pool" => self.pool = pols()?,
            _ => return Err(format!("unknown plan option {key}")),
        }
        Ok(())
    }
}

/// What the last decision did (for the debug panel and the watch tool).
#[derive(Clone, Debug, Default)]
pub struct PlanLog {
    pub time: f64,
    pub wait_value: f64,
    pub value: f64,
    pub kind: Option<Kind>,
    pub continued: bool,
    pub target: Option<u8>,
    pub stage: Option<u8>,
    pub candidates: usize,
    pub rollouts: usize,
}

pub struct Planner {
    pub label: String,
    pub p: PlanParams,
    me: u8,
    next: f64,
    active: Option<Plan>,
    st: State,
    acts: Vec<Action>,
    opp: Vec<PolicyParams>,
    phase: Vec<f64>,
    rng: Rng,
    reinf: Vec<(u8, u8, f64)>,
    pub last: PlanLog,
    pub total_rollouts: u64,
    pub decisions: u64,
}

/// Play `plan` forward from `base` against one opponent sample. None if an order due now is illegal.
#[allow(clippy::too_many_arguments)]
fn rollout(w: &World, p: &PlanParams, me: u8, base: &State, plan: &Plan, opp: &PolicyParams, phase: f64, st: &mut State, acts: &mut Vec<Action>) -> Option<f64> {
    st.copy_from(base);
    let t0 = st.time;
    let end = t0 + p.horizon;
    let nops = plan.ops.len().min(32);
    let mut fired = 0u32;
    for k in 0..nops {
        let op = plan.ops[k];
        if op.at <= t0 + 1e-6 {
            if !st.apply(w, me, op.act) {
                return None;
            }
            fired |= 1 << k;
        }
    }
    let others = st.alive() & !(1u128 << me);
    let mut next_opp = t0 + phase;
    let mut next_me = t0 + p.policy_period;
    let (mut ci, mut acc) = (0usize, 0.0f64);
    let mut steps = 0u32;
    while st.time < end - 1e-9 {
        if (fired.count_ones() as usize) < nops {
            for k in 0..nops {
                if fired >> k & 1 == 0 && plan.ops[k].at <= st.time + 1e-6 {
                    st.apply(w, me, plan.ops[k].act);
                    fired |= 1 << k;
                }
            }
        }
        if st.time >= next_opp {
            next_opp += p.policy_period;
            let mut m = others;
            while m != 0 {
                let o = m.trailing_zeros() as u8;
                m &= m - 1;
                acts.clear();
                policy::act(w, st, o, opp, acts);
                for &a in acts.iter() {
                    st.apply(w, o, a);
                }
            }
        }
        if st.time >= next_me {
            next_me += p.policy_period;
            let mut lock = 0u64;
            for k in 0..nops {
                if fired >> k & 1 == 0 {
                    lock |= 1u64 << Plan::src(plan.ops[k].act);
                }
            }
            acts.clear();
            policy::act_locked(w, st, me, &p.self_model, acts, lock);
            for &a in acts.iter() {
                st.apply(w, me, a);
            }
        }
        st.step(w, p.dt);
        steps += 1;
        if steps % 4 == 0 {
            let alive = st.alive();
            let rest: f64 = CHECK_W[ci..].iter().sum();
            // decided inside the horizon: a sooner win (or later loss) scores better
            if alive >> me & 1 == 0 {
                return Some(acc + rest * (-1.0 - 0.5 * (end - st.time) / p.horizon));
            }
            if alive & others == 0 {
                return Some(acc + rest * (1.0 + 0.5 * (end - st.time) / p.horizon));
            }
        }
        while ci < 4 && st.time >= t0 + CHECK_AT[ci] * p.horizon - 1e-9 {
            acc += CHECK_W[ci] * evaluate(w, st, me, p.kg);
            ci += 1;
        }
    }
    while ci < 4 {
        acc += CHECK_W[ci] * evaluate(w, st, me, p.kg);
        ci += 1;
    }
    Some(acc)
}

#[inline]
fn eta(w: &World, a: usize, b: usize) -> f64 {
    ((w.map.d(a, b) - ARRIVE_SLOP).max(0.0)) / w.rules.orb_speed
}

impl Planner {
    pub fn new(label: &str, p: PlanParams) -> Planner {
        Planner {
            label: label.to_string(),
            p,
            me: 0,
            next: 0.0,
            active: None,
            st: State::empty(0),
            acts: Vec::with_capacity(64),
            opp: Vec::new(),
            phase: Vec::new(),
            rng: Rng::new(1),
            reinf: Vec::new(),
            last: PlanLog::default(),
            total_rollouts: 0,
            decisions: 0,
        }
    }

    /// Which owner this planner plays (the game sets it once; the arena through `Agent::reset`).
    pub fn set_owner(&mut self, me: u8) {
        self.me = me;
        self.active = None;
        self.reinf.clear();
        self.rng = Rng::new(0x5EED ^ me as u64);
    }

    /// Fresh opponent samples for this decision. Every candidate faces the same ones.
    fn sample_opponents(&mut self) {
        self.opp.clear();
        self.phase.clear();
        let n = self.p.samples.max(self.p.screen);
        for j in 0..n {
            let base = &self.p.pool[j % self.p.pool.len()];
            let mut v = base.to_vec();
            if j > 0 && self.p.noise > 0.0 {
                for x in v.iter_mut() {
                    *x *= 1.0 + self.p.noise * (2.0 * self.rng.f64() - 1.0);
                }
            }
            self.opp.push(base.with_vec(&v));
            self.phase.push(0.1 + 0.6 * self.rng.f64());
        }
    }

    /// Value of a plan over opponent samples `0..n`: the mean blended toward the worst.
    fn value(&mut self, w: &World, base: &State, plan: &Plan, n: usize) -> Option<f64> {
        let (mut sum, mut worst) = (0.0, f64::MAX);
        let n = n.min(self.opp.len());
        for j in 0..n {
            let v = rollout(w, &self.p, self.me, base, plan, &self.opp[j], self.phase[j], &mut self.st, &mut self.acts)?;
            self.total_rollouts += 1;
            self.last.rollouts += 1;
            sum += v;
            worst = worst.min(v);
        }
        let mean = sum / n as f64;
        Some(mean + self.p.pessimism * (worst - mean))
    }

    fn reversed_recently(&self, src: u8, dst: u8, now: f64) -> bool {
        self.reinf.iter().any(|&(a, b, t)| a == dst && b == src && now - t < REINF_COOLDOWN)
    }

    /// Single immediate orders: attacks (everything / just enough), reinforcement, invest, upgrade.
    fn direct_orders(&self, w: &World, s: &State, v: &View, lock: u64, out: &mut Vec<Action>) {
        let me = self.me;
        let mut m = v.mine & !lock;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            if s.can_invest(w, me, i) {
                out.push(Action::Invest(i as u8));
            }
            if s.can_upgrade(w, me, i) && s.troops[i] >= s.cap(i) - 10.0 {
                out.push(Action::Upgrade(i as u8));
            }
            let have = s.troops[i].floor();
            if have < 2.0 {
                continue;
            }
            let mut t = w.map.adj_mask[i] & !v.mine;
            while t != 0 {
                let j = t.trailing_zeros() as usize;
                t &= t - 1;
                out.push(Action::Send { src: i as u8, dst: j as u8, amount: have });
                let (po, pt, _) = policy::at_arrival(w, s, v, i, j);
                if po != me {
                    let need = (need_to_take(pt, s.tile[j], s.def_mult(w, j)) * 1.2 + 2.0).ceil();
                    if need < have * 0.8 {
                        out.push(Action::Send { src: i as u8, dst: j as u8, amount: need });
                    }
                }
            }
            if have >= 4.0 {
                let comp = State::flood(w, v.mine, i) & !(1u64 << i);
                let mut dest = comp & (v.border | v.hostile_inbound);
                let mut picked = 0;
                while dest != 0 && picked < 3 {
                    let (mut best, mut bd) = (usize::MAX, f64::MAX);
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
                    if !self.reversed_recently(i as u8, best as u8, s.time) {
                        out.push(Action::Send { src: i as u8, dst: best as u8, amount: have });
                    }
                }
            }
        }
    }

    /// Strike plans: stage troops next to a target, optionally wait, then hit it in one orb.
    fn strikes(&self, w: &World, s: &State, v: &View, out: &mut Vec<(f64, Plan)>) {
        let me = self.me;
        let now = s.time;
        let mut targets = 0u64;
        let mut m = v.mine;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            targets |= w.map.adj_mask[i] & !v.mine;
        }
        while targets != 0 {
            let t = targets.trailing_zeros() as usize;
            targets &= targets - 1;
            // stages: my states bordering the target, best two by reachable force
            let mut stages: Vec<(f64, usize, Vec<usize>)> = Vec::new();
            let mut sm = v.mine & w.map.adj_mask[t];
            while sm != 0 {
                let st = sm.trailing_zeros() as usize;
                sm &= sm - 1;
                let comp = State::flood(w, v.mine, st) & !(1u64 << st);
                let mut feeders: Vec<usize> = Vec::new();
                let mut c = comp & !v.hostile_inbound;
                while c != 0 {
                    let k = c.trailing_zeros() as usize;
                    c &= c - 1;
                    if s.troops[k] >= 4.0 && eta(w, k, st) <= self.p.feed_eta {
                        feeders.push(k);
                    }
                }
                feeders.sort_by(|&a, &b| w.map.d(a, st).partial_cmp(&w.map.d(b, st)).unwrap());
                feeders.truncate(self.p.max_feeders);
                let force = s.troops[st] + feeders.iter().map(|&k| s.troops[k].floor()).sum::<f64>();
                stages.push((force, st, feeders));
            }
            stages.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            stages.truncate(2);
            for (_, st, feeders) in stages {
                let near: Vec<usize> = feeders.iter().copied().take(2).collect();
                let mut sets: Vec<Vec<usize>> = vec![Vec::new(), near.clone()];
                if feeders.len() > near.len() {
                    sets.push(feeders.clone());
                }
                if near.is_empty() {
                    sets.truncate(1);
                }
                for set in &sets {
                    let arrive = set.iter().map(|&k| eta(w, k, st)).fold(0.0f64, f64::max);
                    let pool: f64 = s.troops[st] + set.iter().map(|&k| s.troops[k].floor()).sum::<f64>();
                    for &wait in &self.p.waits {
                        if set.is_empty() && wait <= 0.0 {
                            continue; // that is a direct attack
                        }
                        let at = now + arrive + wait + 0.05;
                        let est = pool + s.growth(w, st) * (arrive + wait);
                        let (po, pt) = s.project(w, t, at - now + eta(w, st, t));
                        if po == me {
                            continue;
                        }
                        let need = need_to_take(pt, s.tile[t], s.def_mult(w, t));
                        if est < 0.85 * need {
                            continue;
                        }
                        let mut ops: Vec<Op> = set.iter().map(|&k| Op { at: now, act: Action::Send { src: k as u8, dst: st as u8, amount: ALL } }).collect();
                        ops.push(Op { at, act: Action::Send { src: st as u8, dst: t as u8, amount: ALL } });
                        out.push((est / need, Plan { kind: Kind::Strike, ops, target: Some(t as u8), stage: Some(st as u8) }));
                    }
                }
            }
        }
    }

    /// What is left of the current plan, if its premises still hold.
    fn incumbent(&mut self, s: &State) -> Option<Plan> {
        let plan = self.active.take()?;
        if plan.ops.is_empty() {
            return None;
        }
        if plan.ops.iter().any(|op| s.owner[Plan::src(op.act) as usize] != self.me) {
            return None; // a reserved state was lost
        }
        if let Some(t) = plan.target {
            if s.owner[t as usize] == self.me {
                return None; // already taken
            }
        }
        Some(plan)
    }

    /// Choose this instant's orders for `me` from position `s`.
    pub fn decide(&mut self, w: &World, s: &State, out: &mut Vec<Action>) {
        let me = self.me;
        let now = s.time;
        self.decisions += 1;
        self.last = PlanLog { time: now, ..Default::default() };
        self.reinf.retain(|&(_, _, t)| now - t < REINF_COOLDOWN);
        let v = View::new(w, s, me);
        if v.mine == 0 {
            self.active = None;
            return;
        }
        self.sample_opponents();
        let inc = self.incumbent(s);

        // 1. candidates, in priority order (kept first when `max_cands` truncates)
        let mut cands: Vec<Plan> = vec![Plan::wait()];
        let inc_idx = inc.map(|p| {
            cands.push(p);
            cands.len() - 1
        });
        for k in 0..self.p.portfolio.len() {
            self.acts.clear();
            let pp = self.p.portfolio[k].clone();
            policy::act(w, s, me, &pp, &mut self.acts);
            let ops: Vec<Op> = self
                .acts
                .iter()
                .filter(|&&a| match a {
                    Action::Send { src, dst, .. } => !(s.owner[dst as usize] == me && self.reversed_recently(src, dst, now)),
                    _ => true,
                })
                .map(|&a| Op { at: now, act: a })
                .collect();
            if !ops.is_empty() {
                cands.push(Plan { kind: Kind::Script, ops, target: None, stage: None });
            }
        }
        let mut strikes = Vec::new();
        self.strikes(w, s, &v, &mut strikes);
        strikes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        cands.extend(strikes.into_iter().map(|x| x.1));
        let mut direct = Vec::new();
        self.direct_orders(w, s, &v, 0, &mut direct);
        cands.extend(direct.iter().map(|&a| Plan { kind: Kind::Direct, ops: vec![Op { at: now, act: a }], target: None, stage: None }));
        if self.p.max_cands > 0 && cands.len() > self.p.max_cands {
            cands.truncate(self.p.max_cands);
        }
        self.last.candidates = cands.len();

        // 2. screen every candidate against the first few opponent samples
        let screen = self.p.screen;
        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(cands.len());
        for (k, c) in cands.iter().enumerate() {
            if let Some(val) = self.value(w, s, c, screen) {
                scored.push((val, k));
            }
        }
        let wait_screen = scored.iter().find(|x| x.1 == 0).map(|x| x.0).unwrap_or(0.0);
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // 3. full comparison of the best few (waiting and the current plan always included)
        let mut finalists: Vec<usize> = scored.iter().take(self.p.final_k).map(|x| x.1).collect();
        for must in [Some(0usize), inc_idx].into_iter().flatten() {
            if !finalists.contains(&must) && scored.iter().any(|x| x.1 == must) {
                finalists.push(must);
            }
        }
        // the same strike (same stage and target) with different timing still counts as continuing
        let inc_key = inc_idx.and_then(|i| cands[i].target.zip(cands[i].stage));
        let continues = |k: usize| Some(k) == inc_idx || (inc_key.is_some() && cands[k].kind == Kind::Strike && cands[k].target.zip(cands[k].stage) == inc_key);
        let mut finals: Vec<(f64, usize)> = Vec::new();
        let mut wait_v = wait_screen;
        for &k in &finalists {
            let c = cands[k].clone();
            if let Some(mut val) = self.value(w, s, &c, self.p.samples) {
                if k == 0 {
                    wait_v = val;
                } else if continues(k) {
                    val += self.p.commit_bonus;
                }
                finals.push((val, k));
            }
        }
        self.last.wait_value = wait_v;
        // a new plan has to beat waiting by `min_gain`; continuing the current one does not
        finals.retain(|&(val, k)| k == 0 || continues(k) || val >= wait_v + self.p.min_gain);
        finals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let (best_v, mut pick) = finals.first().copied().unwrap_or((wait_v, 0));
        if self.p.temperature > 0.0 {
            let near: Vec<usize> = finals.iter().filter(|x| x.0 >= best_v - self.p.temperature).map(|x| x.1).collect();
            if !near.is_empty() {
                pick = near[self.rng.below(near.len())];
            }
        }
        let mut plan = cands[pick].clone();
        self.last.value = best_v;
        self.last.kind = Some(plan.kind);
        self.last.continued = continues(pick);
        self.last.target = plan.target;
        self.last.stage = plan.stage;

        // 4. refinement: single extra orders from unreserved states, kept only if they help
        let plan_v = scored.iter().find(|x| x.1 == pick).map(|x| x.0);
        if let Some(mut cur) = plan_v {
            for _ in 0..self.p.refine {
                let mut lock = 0u64;
                for op in &plan.ops {
                    lock |= 1u64 << Plan::src(op.act);
                }
                let mut extra = Vec::new();
                self.direct_orders(w, s, &v, lock, &mut extra);
                let mut best: Option<(f64, Action)> = None;
                for a in extra {
                    let mut trial = plan.clone();
                    trial.ops.insert(0, Op { at: now, act: a });
                    if let Some(val) = self.value(w, s, &trial, screen) {
                        if val > cur + self.p.min_gain && best.map_or(true, |b| val > b.0) {
                            best = Some((val, a));
                        }
                    }
                }
                match best {
                    Some((val, a)) => {
                        plan.ops.insert(0, Op { at: now, act: a });
                        cur = val;
                    }
                    None => break,
                }
            }
        }

        // 5. fire what is due; keep the rest as the plan to continue
        let due_by = now + 0.05;
        let mut rest = Vec::new();
        for op in plan.ops.drain(..) {
            if op.at <= due_by {
                out.push(op.act);
                if let Action::Send { src, dst, .. } = op.act {
                    if s.owner[dst as usize] == me {
                        self.reinf.push((src, dst, now));
                    }
                }
            } else {
                rest.push(op);
            }
        }
        plan.ops = rest;
        self.active = if plan.ops.is_empty() { None } else { Some(plan) };
    }
}

impl Agent for Planner {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn reset(&mut self, _w: &World, me: u8, seed: u64) {
        self.set_owner(me);
        self.rng = Rng::new(seed ^ 0xA5A5);
        self.next = 0.1 + (seed % 5) as f64 * 0.02;
    }
    fn act(&mut self, w: &World, s: &State, out: &mut Vec<Action>) {
        if s.time < self.next {
            return;
        }
        self.next = s.time + self.p.cadence;
        self.decide(w, s, out);
    }
    fn stats(&self) -> (u64, u64) {
        (self.decisions, self.total_rollouts)
    }
    fn log(&self) -> Option<String> {
        let l = &self.last;
        l.kind.map(|k| {
            format!(
                "{:?}{}{}{} value {:+.4} (wait {:+.4}) cands {} rollouts {}",
                k,
                if l.continued { " (continued)" } else { "" },
                l.stage.map_or(String::new(), |x| format!(" stage {x}")),
                l.target.map_or(String::new(), |x| format!(" target {x}")),
                l.value,
                l.wait_value,
                l.candidates,
                l.rollouts
            )
        })
    }
}
