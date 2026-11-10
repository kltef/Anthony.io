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
    Hold,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub kind: Kind,
    /// Orders in firing order.
    pub ops: Vec<Op>,
    pub target: Option<u8>,
    pub stage: Option<u8>,
    /// States this plan keeps where they are (a Hold plan: the stack stays as a deterrent), and
    /// until when.
    pub hold: u64,
    pub until: f64,
    /// A chain's first capture: (state, time it should be mine by). Orders from it are not
    /// "a reserved state was lost" while that capture is still on its way.
    pub via: Option<(u8, f64)>,
}

impl Plan {
    fn wait() -> Plan {
        Plan::new(Kind::Wait, Vec::new(), None, None)
    }
    fn new(kind: Kind, ops: Vec<Op>, target: Option<u8>, stage: Option<u8>) -> Plan {
        Plan { kind, ops, target, stage, hold: 0, until: 0.0, via: None }
    }
    fn src(a: Action) -> u8 {
        match a {
            Action::Send { src, .. } => src,
            Action::Invest(i) | Action::Upgrade(i) => i,
        }
    }
}

/// How a simulated opponent plays: a scripted style, or the play-tester-style bot (masses into
/// hubs, only hits what is thin), which the scripted styles never do.
#[derive(Clone, Debug)]
pub enum Opp {
    Policy(PolicyParams),
    Chris(crate::chris::ChrisParams),
    /// The play-tester's funnel (one roaming spearhead fed by every other state).
    Funnel(crate::chris::ChrisParams),
}

impl Opp {
    fn act(&self, w: &World, s: &State, me: u8, out: &mut Vec<Action>) {
        match self {
            Opp::Policy(p) => policy::act(w, s, me, p, out),
            Opp::Chris(p) => crate::chris::act(w, s, me, p, out),
            Opp::Funnel(p) => crate::chris::act_funnel_stateless(w, s, me, p, out),
        }
    }
    /// A randomised copy (relative perturbation of the numeric parameters).
    fn perturbed(&self, noise: f64, rng: &mut Rng) -> Opp {
        let mut jit = |x: f64| x * (1.0 + noise * (2.0 * rng.f64() - 1.0));
        match self {
            Opp::Policy(p) => {
                let v: Vec<f64> = p.to_vec().into_iter().map(&mut jit).collect();
                Opp::Policy(p.with_vec(&v))
            }
            Opp::Chris(c) => Opp::Chris(crate::chris::ChrisParams {
                neutral_margin: jit(c.neutral_margin).max(1.0),
                enemy_ratio: jit(c.enemy_ratio).max(1.0),
                min_feed: jit(c.min_feed),
                ..c.clone()
            }),
            Opp::Funnel(c) => Opp::Funnel(crate::chris::ChrisParams {
                neutral_margin: jit(c.neutral_margin).max(1.0),
                enemy_ratio: jit(c.enemy_ratio).max(1.0),
                ..c.clone()
            }),
        }
    }
}

/// The opponent model fitted to the play-tester (see chris.rs; settings from the arena fit).
pub fn chris_model() -> crate::chris::ChrisParams {
    crate::chris::ChrisParams { enemy_ratio: 3.0, neutral_margin: 2.0, min_feed: 4.0, ..Default::default() }
}

/// The funnel he won with on Nov 9: one spearhead, everything else swiped into it every second.
pub fn funnel_model() -> crate::chris::ChrisParams {
    crate::chris::ChrisParams { funnel: true, min_feed: 1.0, neutral_margin: 1.2, enemy_ratio: 3.0, ..Default::default() }
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
    pub pool: Vec<Opp>,
    /// Never consider an attack that cannot take its target on arrival (it only feeds the
    /// defender: under Lanchester combat a small wave into a big garrison dies for almost nothing).
    pub no_hopeless: bool,
    /// Also offer each strike sending only what the capture needs (plus a margin), keeping the rest
    /// home, next to the all-in version.
    pub strike_partial: bool,
    /// Unpredictability: when starting a NEW plan, pick at random among new plans within this much
    /// of the best (a plan already under way is never dropped for this).
    pub vary: f64,
    /// Unpredictability: random extra seconds (0..jitter) added to each strike wait, per decision.
    pub jitter: f64,
    /// Seconds to keep the same opponent samples across decisions (0 = fresh every decision).
    /// Fast cadences re-score plans many times a second; reusing samples stops pure sampling
    /// noise from flipping the choice.
    pub resample: f64,
    /// Anti-dither: for this many seconds after a big stack (`settle_min`+ troops) lands on one
    /// of my states from another, it may strike but not be moved on to another of my states, and
    /// no move (strike feeders included) may send troops straight back the way they came.
    /// Against a roaming enemy spearhead the planner otherwise re-picks its staging state every
    /// decision and ferries its army back and forth (the play-tester's Nov 9 wins). 0 = off.
    pub settle: f64,
    pub settle_min: f64,
    /// Also park a big stack that just arrived (no onward own-to-own move for `settle` s). Off =
    /// only the straight-back move is blocked (strike feeders included).
    pub settle_park: bool,
    /// Also block loops of any length: no own-to-own move onto a state that itself sent troops on
    /// to another of my states within `settle` seconds (A->B->C->A passes the straight-back check;
    /// at the 4x rhythm Nightmare+ cycled stacks Vermont->Massachusetts->Connecticut->Vermont
    /// several times a second, play-tester's Nov 9 20:38 win).
    pub settle_loop: bool,
    /// Also consider holding my biggest stacks where they are (deterrence) as a plan.
    pub hold: bool,
    /// Chains: the best N strikes may roll on from the captured state into a second target.
    pub chain: usize,
    /// Two-front plans: pair the best N strikes that use disjoint states into one plan.
    pub fronts: usize,
    /// Evacuation: a state about to fall sends its troops to a safe state of mine instead.
    pub evac: bool,
    /// Anti-stall: after this many seconds without gaining a state while the strongest opponent is
    /// stronger, judge plans on the average outcome (no worst-case blend) and take a capture plan
    /// over waiting if it is within `stall_slack` of it. A cornered AI facing a huge army otherwise
    /// waits forever, since over the horizon every expansion looks punishable (the play-tester's
    /// Nov 9 20:30 win: 6 minutes in New Hampshire with 200 troops next to two 53-troop neutrals).
    /// 0 = off.
    pub stall: f64,
    pub stall_slack: f64,
    /// Human input limits (the Nightmare tier). Measured from the play-tester's recorded games: he
    /// sweeps up to ~10 states in half a second, but always into one or two targets with one
    /// gesture, and sustains at most ~4.5 sends a second. With this on, orders over a limit wait
    /// for the next decision, and amounts snap to the Send button's All / Half / Quarter.
    pub human: bool,
    /// Most distinct targets (a send's target, or the state tapped to invest / upgrade) in 0.5 s.
    pub h_targets: usize,
    /// Most sends in 0.5 s, in 1 s and in 10 s.
    pub h_sends: [usize; 3],
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
            pool: vec![
                Opp::Policy(tuned()),
                Opp::Chris(chris_model()),
                Opp::Policy(PolicyParams::rush()),
                Opp::Policy(PolicyParams::farmer()),
                Opp::Policy(PolicyParams::default()),
                Opp::Policy(PolicyParams::economist()),
            ],
            no_hopeless: true,
            strike_partial: true,
            vary: 0.0,
            jitter: 0.0,
            resample: 0.0,
            settle: 0.0,
            settle_min: 20.0,
            settle_park: true,
            settle_loop: false,
            hold: false,
            chain: 0,
            fronts: 0,
            evac: false,
            stall: 0.0,
            stall_slack: 0.05,
            human: false,
            h_targets: 2,
            h_sends: [7, 10, 45],
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
            "pool" => {
                self.pool = val
                    .split('+')
                    .map(|n| match n {
                        "chris" => Ok(Opp::Chris(chris_model())),
                        "funnel" => Ok(Opp::Funnel(funnel_model())),
                        _ => crate::roster::policy(n).map(Opp::Policy).ok_or(format!("unknown policy {n}")),
                    })
                    .collect::<Result<_, _>>()?
            }
            "hopeless" => self.no_hopeless = num()? == 0.0,
            "partial" => self.strike_partial = num()? != 0.0,
            "vary" => self.vary = num()?,
            "jitter" => self.jitter = num()?,
            "resample" => self.resample = num()?,
            "settle" => self.settle = num()?,
            "settle_min" => self.settle_min = num()?,
            "park" => self.settle_park = num()? != 0.0,
            "loop" => self.settle_loop = num()? != 0.0,
            "hold" => self.hold = num()? != 0.0,
            "chain" => self.chain = num()? as usize,
            "fronts" => self.fronts = num()? as usize,
            "evac" => self.evac = num()? != 0.0,
            "stall" => self.stall = num()?,
            "stall_slack" => self.stall_slack = num()?,
            "human" => self.human = num()? != 0.0,
            "h_targets" => self.h_targets = num()?.max(1.0) as usize,
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
    opp: Vec<Opp>,
    phase: Vec<f64>,
    rng: Rng,
    reinf: Vec<(u8, u8, f64)>,
    /// (state, time) where a big stack from another of my states recently landed (anti-dither).
    settled: Vec<(u8, f64)>,
    /// anti-stall bookkeeping: most states held so far, when that was reached, and this
    /// decision's worst-case weight
    owned_best: u32,
    gained_at: f64,
    pess_now: f64,
    waits_now: Vec<f64>,
    sampled_at: f64,
    /// (time, target) of recent orders, for the human input limits.
    fired: Vec<(f64, u16)>,
    pub last: PlanLog,
    pub total_rollouts: u64,
    pub decisions: u64,
}

/// Play `plan` forward from `base` against one opponent sample. None if an order due now is illegal.
#[allow(clippy::too_many_arguments)]
fn rollout(w: &World, p: &PlanParams, me: u8, base: &State, plan: &Plan, opp: &Opp, phase: f64, st: &mut State, acts: &mut Vec<Action>) -> Option<f64> {
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
                opp.act(w, st, o, acts);
                for &a in acts.iter() {
                    st.apply(w, o, a);
                }
            }
        }
        if st.time >= next_me {
            next_me += p.policy_period;
            let mut lock = if st.time < plan.until { plan.hold } else { 0 };
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
            settled: Vec::new(),
            owned_best: 0,
            gained_at: 0.0,
            pess_now: 0.5,
            waits_now: Vec::new(),
            sampled_at: f64::NEG_INFINITY,
            fired: Vec::new(),
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
        self.settled.clear();
        self.owned_best = 0;
        self.gained_at = 0.0;
        self.sampled_at = f64::NEG_INFINITY;   // a new match draws fresh opponent samples
        self.fired.clear();
        self.rng = Rng::new(0x5EED ^ me as u64);
    }

    /// Fresh opponent samples for this decision. Every candidate faces the same ones.
    fn sample_opponents(&mut self) {
        self.opp.clear();
        self.phase.clear();
        let n = self.p.samples.max(self.p.screen);
        for j in 0..n {
            let base = self.p.pool[j % self.p.pool.len()].clone();
            let o = if j >= self.p.pool.len() && self.p.noise > 0.0 { base.perturbed(self.p.noise, &mut self.rng) } else { base };
            self.opp.push(o);
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
        Some(mean + self.pess_now * (worst - mean))
    }

    fn reversed_recently(&self, src: u8, dst: u8, now: f64) -> bool {
        self.reinf.iter().any(|&(a, b, t)| a == dst && b == src && now - t < REINF_COOLDOWN)
    }

    /// May `src` send troops on to my own state `dst` now? (see `PlanParams::settle`)
    fn own_move_ok(&self, src: u8, dst: u8, now: f64) -> bool {
        if self.p.settle <= 0.0 {
            return !self.reversed_recently(src, dst, now);
        }
        let back = self.reinf.iter().any(|&(a, b, t)| a == dst && (b == src || self.p.settle_loop) && now - t < self.p.settle.max(REINF_COOLDOWN));
        let parked = self.p.settle_park && self.settled.iter().any(|&(k, t)| k == src && now - t < self.p.settle);
        !back && !parked
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
            if self.p.evac && v.hostile_inbound >> i & 1 == 1 && have >= 6.0 {
                // will it fall to what is already flying at it?
                let mut last = 0.0f64;
                for o in &s.orbs {
                    if o.dst as usize == i && o.owner != me {
                        last = last.max(o.rem / w.rules.orb_speed);
                    }
                }
                if s.project(w, i, last + 0.1).0 != me {
                    // nearest of my states that nothing hostile is flying at
                    let comp = State::flood(w, v.mine, i) & !(1u64 << i) & !v.hostile_inbound;
                    let (mut best, mut bd) = (usize::MAX, f64::MAX);
                    let mut c = comp;
                    while c != 0 {
                        let k = c.trailing_zeros() as usize;
                        c &= c - 1;
                        if w.map.d(i, k) < bd {
                            bd = w.map.d(i, k);
                            best = k;
                        }
                    }
                    if best != usize::MAX {
                        out.push(Action::Send { src: i as u8, dst: best as u8, amount: have });
                    }
                }
            }
            let mut t = w.map.adj_mask[i] & !v.mine;
            while t != 0 {
                let j = t.trailing_zeros() as usize;
                t &= t - 1;
                let (po, pt, _) = policy::at_arrival(w, s, v, i, j);
                if po == me {
                    continue; // already being taken by troops in flight
                }
                let take = need_to_take(pt, s.tile[j], s.def_mult(w, j));
                // a wave that cannot take the target only feeds the defender
                if self.p.no_hopeless && have < take {
                    continue;
                }
                out.push(Action::Send { src: i as u8, dst: j as u8, amount: have });
                let need = (take * 1.2 + 2.0).ceil();
                if need < have * 0.8 {
                    out.push(Action::Send { src: i as u8, dst: j as u8, amount: need });
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
                    if self.own_move_ok(i as u8, best as u8, s.time) {
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
                    if s.troops[k] >= 4.0 && eta(w, k, st) <= self.p.feed_eta && (self.p.settle <= 0.0 || self.own_move_ok(k as u8, st as u8, now)) {
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
                    for &wait in &self.waits_now {
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
                        if est < (if self.p.no_hopeless { 1.0 } else { 0.85 }) * need {
                            continue;
                        }
                        let feed: Vec<Op> = set.iter().map(|&k| Op { at: now, act: Action::Send { src: k as u8, dst: st as u8, amount: ALL } }).collect();
                        let mut ops = feed.clone();
                        ops.push(Op { at, act: Action::Send { src: st as u8, dst: t as u8, amount: ALL } });
                        out.push((est / need, Plan::new(Kind::Strike, ops, Some(t as u8), Some(st as u8))));
                        // the same strike sending only what the capture needs plus a margin
                        let part = (need * 1.3 + 4.0).ceil();
                        if self.p.strike_partial && part < 0.8 * est {
                            let mut ops = feed;
                            ops.push(Op { at, act: Action::Send { src: st as u8, dst: t as u8, amount: part } });
                            out.push((est / need, Plan::new(Kind::Strike, ops, Some(t as u8), Some(st as u8))));
                        }
                    }
                }
            }
        }
    }

    /// Chains: after `first` (an all-in strike) captures its target, roll the surviving stack on
    /// into a neighbouring second target.
    fn chains(&self, w: &World, s: &State, v: &View, first: &Plan, out: &mut Vec<(f64, Plan)>) {
        let me = self.me;
        let (t, st) = match (first.target, first.stage) {
            (Some(t), Some(st)) => (t as usize, st as usize),
            _ => return,
        };
        let strike_at = first.ops.last().unwrap().at;
        let pool: f64 = s.troops[st] + first.ops[..first.ops.len() - 1].iter().map(|o| s.troops[Plan::src(o.act) as usize].floor()).sum::<f64>();
        let est = pool + s.growth(w, st) * (strike_at - s.time);
        let land1 = strike_at + eta(w, st, t);
        let (po, pt) = s.project(w, t, land1 - s.time);
        if po == me {
            return;
        }
        let (o1, left) = land_calc(w.rules.depth, po, pt, s.tile[t], s.def_mult(w, t), me, est);
        if o1 != me || left < 8.0 {
            return;
        }
        let mut t2s = w.map.adj_mask[t] & !v.mine & !(1u64 << t);
        let mut best: Vec<(f64, usize)> = Vec::new();
        while t2s != 0 {
            let j = t2s.trailing_zeros() as usize;
            t2s &= t2s - 1;
            let go = land1 + 0.3;
            let (qo, qt) = s.project(w, j, go - s.time + eta(w, t, j));
            if qo == me {
                continue;
            }
            let need = need_to_take(qt, s.tile[j], s.def_mult(w, j));
            if left >= need {
                best.push((left / need, j));
            }
        }
        best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(r, j) in best.iter().take(2) {
            let mut ops = first.ops.clone();
            ops.push(Op { at: land1 + 0.3, act: Action::Send { src: t as u8, dst: j as u8, amount: ALL } });
            let mut pl = Plan::new(Kind::Strike, ops, Some(j as u8), Some(st as u8));
            pl.via = Some((t as u8, land1 + 1.0));
            out.push((r, pl));
        }
    }

    /// Human input limits: None if `act` would exceed them now (it waits for the next decision),
    /// otherwise the order with its amount snapped to All / Half / Quarter.
    fn humanize(&mut self, s: &State, now: f64, act: Action) -> Option<Action> {
        self.fired.retain(|&(t, _)| now - t < 10.0);
        let (target, is_send) = match act {
            Action::Send { dst, .. } => (dst as u16, true),
            Action::Invest(i) | Action::Upgrade(i) => (256 + i as u16, false),
        };
        let recent = |w: f64| self.fired.iter().filter(move |&&(t, _)| now - t < w);
        let mut targets: Vec<u16> = recent(0.5).map(|&(_, x)| x).collect();
        targets.sort_unstable();
        targets.dedup();
        if !targets.contains(&target) && targets.len() >= self.p.h_targets {
            return None;
        }
        if is_send {
            let sends = |w: f64| recent(w).filter(|&&(_, x)| x < 256).count();
            if sends(0.5) >= self.p.h_sends[0] || sends(1.0) >= self.p.h_sends[1] || sends(10.0) >= self.p.h_sends[2] {
                return None;
            }
        }
        let out = match act {
            Action::Send { src, dst, amount } => {
                let have = s.troops[src as usize].floor();
                // the smallest button that still sends at least what the plan wants
                let amt = if amount >= have {
                    ALL
                } else if amount <= (have / 4.0).floor() && (have / 4.0).floor() >= 1.0 {
                    (have / 4.0).floor()
                } else if amount <= (have / 2.0).floor() && (have / 2.0).floor() >= 1.0 {
                    (have / 2.0).floor()
                } else {
                    ALL
                };
                Action::Send { src, dst, amount: amt }
            }
            a => a,
        };
        self.fired.push((now, target));
        Some(out)
    }

    /// What is left of the current plan, if its premises still hold.
    fn incumbent(&mut self, s: &State) -> Option<Plan> {
        let plan = self.active.take()?;
        if plan.ops.is_empty() && (plan.hold == 0 || s.time >= plan.until) {
            return None;
        }
        if plan.hold & !s.owned_mask(self.me) != 0 {
            return None; // a held state was lost
        }
        let lost = |src: u8| {
            s.owner[src as usize] != self.me && !matches!(plan.via, Some((v, by)) if v == src && s.time < by)
        };
        if plan.ops.iter().any(|op| lost(Plan::src(op.act))) {
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
        let keep = self.p.settle.max(REINF_COOLDOWN);
        self.reinf.retain(|&(_, _, t)| now - t < keep);
        self.settled.retain(|&(_, t)| now - t < keep);
        let v = View::new(w, s, me);
        if v.mine == 0 {
            self.active = None;
            return;
        }
        let held = v.mine.count_ones();
        if held > self.owned_best {
            self.owned_best = held;
            self.gained_at = now;
        }
        let stalled = self.p.stall > 0.0 && now - self.gained_at > self.p.stall && {
            let mine = crate::agent::strength(w, s, me);
            let mut top = 0.0f64;
            let alive = s.alive();
            for o in 1..=8u8 {
                if o != me && alive >> o & 1 == 1 {
                    top = top.max(crate::agent::strength(w, s, o));
                }
            }
            top > mine
        };
        self.pess_now = if stalled { 0.0 } else { self.p.pessimism };
        let n_need = self.p.samples.max(self.p.screen);
        if self.p.resample <= 0.0 || now - self.sampled_at >= self.p.resample || self.opp.len() != n_need {
            self.sample_opponents();
            self.sampled_at = now;
        }
        self.waits_now = self.p.waits.clone();
        if self.p.jitter > 0.0 {
            for x in self.waits_now.iter_mut() {
                *x += self.p.jitter * self.rng.f64();
            }
        }
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
                    Action::Send { src, dst, .. } => !(s.owner[dst as usize] == me && !self.own_move_ok(src, dst, now)),
                    _ => true,
                })
                .map(|&a| Op { at: now, act: a })
                .collect();
            if !ops.is_empty() {
                cands.push(Plan::new(Kind::Script, ops, None, None));
            }
        }
        let mut strikes = Vec::new();
        self.strikes(w, s, &v, &mut strikes);
        strikes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if self.p.chain > 0 {
            let mut chains = Vec::new();
            for (_, pl) in strikes.iter().filter(|x| x.1.ops.last().map_or(false, |o| matches!(o.act, Action::Send { amount, .. } if amount >= ALL))).take(self.p.chain) {
                self.chains(w, s, &v, pl, &mut chains);
            }
            strikes.extend(chains);
        }
        if self.p.fronts > 1 {
            let top: Vec<Plan> = strikes.iter().take(self.p.fronts).map(|x| x.1.clone()).collect();
            let used = |pl: &Plan| pl.ops.iter().fold(0u64, |m, o| m | 1u64 << Plan::src(o.act));
            for a in 0..top.len() {
                for b in a + 1..top.len() {
                    if top[a].target == top[b].target || used(&top[a]) & used(&top[b]) != 0 {
                        continue;
                    }
                    let mut ops = top[a].ops.clone();
                    ops.extend(top[b].ops.iter().copied());
                    ops.sort_by(|x, y| x.at.partial_cmp(&y.at).unwrap());
                    strikes.push((0.0, Plan::new(Kind::Strike, ops, top[a].target, top[a].stage)));
                }
            }
        }
        cands.extend(strikes.into_iter().map(|x| x.1));
        if self.p.hold {
            // hold my biggest stacks (those big enough to deter) where they are
            let mut big: Vec<(f64, usize)> = Vec::new();
            let mut m = v.mine & v.border;
            while m != 0 {
                let i = m.trailing_zeros() as usize;
                m &= m - 1;
                if s.troops[i] >= self.p.settle_min.max(20.0) {
                    big.push((s.troops[i], i));
                }
            }
            big.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for &(_, i) in big.iter().take(2) {
                let mut pl = Plan::new(Kind::Hold, Vec::new(), None, Some(i as u8));
                pl.hold = 1u64 << i;
                pl.until = now + self.p.horizon.min(10.0);
                cands.push(pl);
            }
        }
        let mut direct = Vec::new();
        self.direct_orders(w, s, &v, 0, &mut direct);
        cands.extend(direct.iter().map(|&a| Plan::new(Kind::Direct, vec![Op { at: now, act: a }], None, None)));
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
        let inc_hold = inc_idx.filter(|&i| cands[i].kind == Kind::Hold).and_then(|i| cands[i].stage);
        let continues = |k: usize| {
            Some(k) == inc_idx
                || (inc_key.is_some() && cands[k].kind == Kind::Strike && cands[k].target.zip(cands[k].stage) == inc_key)
                || (inc_hold.is_some() && cands[k].kind == Kind::Hold && cands[k].stage == inc_hold)
        };
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
        // stalled: the best capture plan within `stall_slack` of waiting beats waiting
        let captures = |pl: &Plan| match pl.kind {
            Kind::Strike => pl.target.map_or(false, |t| s.owner[t as usize] != me),
            Kind::Direct => pl.ops.iter().any(|o| matches!(o.act, Action::Send { dst, .. } if s.owner[dst as usize] != me)),
            _ => false,
        };
        let stall_pick = if stalled {
            finals.iter().filter(|&&(val, k)| k != 0 && captures(&cands[k]) && val >= wait_v - self.p.stall_slack)
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).copied()
        } else {
            None
        };
        // a new plan has to beat waiting by `min_gain`; continuing the current one does not
        finals.retain(|&(val, k)| k == 0 || continues(k) || val >= wait_v + self.p.min_gain);
        finals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let (mut best_v, mut pick) = finals.first().copied().unwrap_or((wait_v, 0));
        if let Some((sv, sk)) = stall_pick {
            if pick == 0 || !captures(&cands[pick]) {
                best_v = sv;
                pick = sk;
            }
        }
        if self.p.temperature > 0.0 {
            let near: Vec<usize> = finals.iter().filter(|x| x.0 >= best_v - self.p.temperature).map(|x| x.1).collect();
            if !near.is_empty() {
                pick = near[self.rng.below(near.len())];
            }
        }
        if self.p.vary > 0.0 && pick != 0 && !continues(pick) {
            let near: Vec<usize> = finals.iter().filter(|x| x.0 >= best_v - self.p.vary && x.1 != 0 && !continues(x.1)).map(|x| x.1).collect();
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
                let mut lock = plan.hold;
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
                let act = if self.p.human {
                    match self.humanize(s, now, op.act) {
                        Some(a) => a,
                        None => {
                            rest.push(op); // over a human limit: try again next decision
                            continue;
                        }
                    }
                } else {
                    op.act
                };
                out.push(act);
                if let Action::Send { src, dst, amount } = act {
                    if s.owner[dst as usize] == me {
                        self.reinf.push((src, dst, now));
                        if amount.min(s.troops[src as usize]).floor() >= self.p.settle_min {
                            self.settled.push((dst, now));
                        }
                    }
                }
            } else {
                rest.push(op);
            }
        }
        plan.ops = rest;
        self.active = if plan.ops.is_empty() && plan.hold == 0 { None } else { Some(plan) };
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
