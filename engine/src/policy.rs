//! The base policy: a fast, deterministic rule-of-thumb player.
//!
//! It is used three ways: as the move generator inside search rollouts (for both sides), as the
//! fallback when there is no time to search, and (with different parameters) as the scripted
//! opponents in the arena league. It uses exact arithmetic from the rules (arrival times, the
//! projected defender at arrival, Lanchester losses) rather than guesses.

use crate::map::MAXN;
use crate::rules::*;

#[derive(Clone, Debug)]
pub struct PolicyParams {
    /// Attack only when the force available is at least `margin` x the minimum needed.
    pub margin: f64,
    /// Minimum net value (troop-equivalents) for an attack to be worth making.
    pub theta: f64,
    /// Seconds of growth a captured state is valued at.
    pub kg: f64,
    /// Multiplier on the value of taking enemy / neutral states.
    pub enemy_bias: f64,
    pub neutral_bias: f64,
    /// Fraction of the strongest adjacent enemy stack to leave behind when attacking elsewhere.
    pub guard: f64,
    /// Highest invest tier the policy will buy (0 = never invest).
    pub invest_max: u8,
    /// Troops to keep after investing.
    pub invest_reserve: f64,
    /// Buy cap upgrades when a safe state is sitting at its cap.
    pub upgrade: bool,
    /// Smallest stack worth forwarding from the interior to the front.
    pub forward_min: f64,
    /// Attack enemy states only when my total troops are at least this multiple of theirs (0 = always).
    pub hoard_ratio: f64,
    /// React to incoming attacks (reinforce or evacuate).
    pub defend: bool,
    /// Cost per second of travel when scoring an attack.
    pub eta_w: f64,
    /// Extra value of a special tile.
    pub tile_bonus: f64,
    /// How far above its cap a border state may be stacked by forwarding.
    pub cap_slack: f64,
    /// How much the attack threshold drops for a stack sitting at its cap.
    pub cap_relief: f64,
}

impl Default for PolicyParams {
    fn default() -> Self {
        PolicyParams {
            margin: 1.15,
            theta: 4.0,
            kg: 45.0,
            enemy_bias: 1.0,
            neutral_bias: 1.0,
            guard: 0.5,
            invest_max: 3,
            invest_reserve: 8.0,
            upgrade: false,
            forward_min: 12.0,
            hoard_ratio: 0.0,
            defend: true,
            eta_w: 1.0,
            tile_bonus: 10.0,
            cap_slack: 25.0,
            cap_relief: 10.0,
        }
    }
}

impl PolicyParams {
    /// All-out early aggression: thin margins, ignores economy.
    pub fn rush() -> Self {
        PolicyParams { margin: 1.05, theta: -5.0, enemy_bias: 2.0, guard: 0.0, invest_max: 0, forward_min: 6.0, ..Default::default() }
    }
    /// Economy first: invests everywhere, attacks only with a large surplus.
    pub fn economist() -> Self {
        PolicyParams { margin: 1.5, theta: 10.0, invest_max: 3, invest_reserve: 2.0, upgrade: true, guard: 0.8, ..Default::default() }
    }
    /// Farm-and-snowball: bank troops, invest, and only strike the enemy at a big total advantage.
    pub fn farmer() -> Self {
        PolicyParams { margin: 1.4, theta: 6.0, invest_max: 3, upgrade: true, guard: 1.0, hoard_ratio: 2.5, enemy_bias: 1.5, ..Default::default() }
    }
    /// Pure defence: never attacks the enemy, takes only cheap neutrals.
    pub fn turtle() -> Self {
        PolicyParams { margin: 1.6, theta: 12.0, invest_max: 3, upgrade: true, guard: 1.2, hoard_ratio: 6.0, ..Default::default() }
    }
}

/// Per-call scratch data about one owner's position.
pub struct View {
    pub me: u8,
    pub mine: u64,
    /// My states with at least one neighbour I do not own.
    pub border: u64,
    /// My states with at least one enemy (non-neutral, not me) neighbour.
    pub contact: u64,
    /// Strongest adjacent enemy stack per state.
    pub enemy_adj_max: [f64; MAXN],
    /// States that have at least one orb heading to them.
    pub inbound: u64,
    /// States that have at least one hostile (to `me`) orb heading to them.
    pub hostile_inbound: u64,
    pub my_troops: f64,
    pub enemy_troops: f64,
}

impl View {
    pub fn new(w: &World, s: &State, me: u8) -> View {
        let mine = s.owned_mask(me);
        let mut v = View { me, mine, border: 0, contact: 0, enemy_adj_max: [0.0; MAXN], inbound: 0, hostile_inbound: 0, my_troops: 0.0, enemy_troops: 0.0 };
        for i in 0..s.n {
            let o = s.owner[i];
            if o == me {
                v.my_troops += s.troops[i];
                let others = w.map.adj_mask[i] & !mine;
                if others != 0 {
                    v.border |= 1u64 << i;
                    let mut m = others;
                    while m != 0 {
                        let j = m.trailing_zeros() as usize;
                        m &= m - 1;
                        if s.owner[j] != NEUTRAL {
                            v.contact |= 1u64 << i;
                            if s.troops[j] > v.enemy_adj_max[i] {
                                v.enemy_adj_max[i] = s.troops[j];
                            }
                        }
                    }
                }
            } else if o != NEUTRAL {
                v.enemy_troops += s.troops[i];
            }
        }
        for o in &s.orbs {
            v.inbound |= 1u64 << o.dst;
            if o.owner == me {
                v.my_troops += o.count;
            } else {
                v.enemy_troops += o.count;
                v.hostile_inbound |= 1u64 << o.dst;
            }
        }
        v
    }
}

/// Where state `j` will stand when an orb sent now from `i` arrives: (owner, troops, eta).
#[inline]
pub fn at_arrival(w: &World, s: &State, v: &View, i: usize, j: usize) -> (u8, f64, f64) {
    let eta = ((w.map.d(i, j) - ARRIVE_SLOP).max(0.0)) / w.rules.orb_speed;
    if v.inbound >> j & 1 == 1 {
        let (o, t) = s.project(w, j, eta);
        (o, t, eta)
    } else {
        // closed form: nothing is flying there
        let o = s.owner[j];
        let mut t = s.troops[j];
        if o != NEUTRAL {
            let cap = s.cap(j);
            if t < cap {
                let mut g = w.rules.grow[o as usize] * tile_grow(s.tile[j]);
                if w.rules.depth {
                    g *= 1.0 + 0.4 * s.inv[j] as f64;
                }
                t = (t + g * eta).min(cap);
            }
        }
        (o, t, eta)
    }
}

/// Value (in troop-equivalents) of capturing state `j` whose projected holder is `owner`.
#[inline]
pub fn capture_gain(w: &World, s: &State, p: &PolicyParams, me: u8, j: usize, owner: u8, troops: f64) -> f64 {
    let tg = tile_grow(s.tile[j]);
    let mine = w.rules.grow[me as usize] * tg;
    let tile_bonus = if s.tile[j] != TILE_NONE { p.tile_bonus } else { 0.0 };
    if owner == NEUTRAL {
        (p.kg * mine + tile_bonus) * p.neutral_bias
    } else {
        let theirs = w.rules.grow[owner as usize] * tg * (1.0 + 0.4 * s.inv[j] as f64);
        (p.kg * (mine + theirs) + troops + tile_bonus) * p.enemy_bias
    }
}

/// Decide this owner's orders for the current instant. Appends to `out`; at most one order per state.
pub fn act(w: &World, s: &State, me: u8, p: &PolicyParams, out: &mut Vec<Action>) {
    let v = View::new(w, s, me);
    if v.mine == 0 {
        return;
    }
    let n = s.n;
    let speed = w.rules.orb_speed;
    let mut used = 0u64; // states that already have an order this call
    let mut claimed = 0u64; // targets already being attacked this call
    let hoarding = p.hoard_ratio > 0.0 && v.my_troops < p.hoard_ratio * v.enemy_troops;

    // 1. Defence: a state that will fall to troops already in flight gets reinforced, or emptied.
    if p.defend && v.hostile_inbound & v.mine != 0 {
        let mut m = v.hostile_inbound & v.mine;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            // look just past the last hostile arrival
            let (mut first, mut last) = (1e9f64, 0.0f64);
            for o in &s.orbs {
                if o.dst as usize == i && o.owner != me {
                    let eta = ((o.rem - ARRIVE_SLOP).max(0.0)) / speed;
                    first = first.min(eta);
                    last = last.max(eta);
                }
            }
            let (po, _) = s.project(w, i, last + 0.05);
            if po == me {
                continue;
            }
            // rescuer: the closest connected stack that arrives before the first blow and holds it
            let comp = State::flood(w, v.mine, i);
            let mut best = usize::MAX;
            let mut best_d = 1e18;
            let mut c = comp & !(1u64 << i) & !used;
            while c != 0 {
                let k = c.trailing_zeros() as usize;
                c &= c - 1;
                let amt = s.troops[k].floor();
                if amt < 2.0 {
                    continue;
                }
                let eta = ((w.map.d(k, i) - ARRIVE_SLOP).max(0.0)) / speed;
                if eta >= first {
                    continue;
                }
                let (ho, _) = s.project_with(w, i, last + 0.05, Some((me, amt, eta)));
                if ho == me && w.map.d(k, i) < best_d {
                    best_d = w.map.d(k, i);
                    best = k;
                }
            }
            if best != usize::MAX {
                out.push(Action::Send { src: best as u8, dst: i as u8, amount: s.troops[best].floor() });
                used |= 1u64 << best;
                continue;
            }
            // cannot be held: get the troops out so the attacker only wins an empty state
            let amt = s.troops[i].floor();
            if amt >= 1.0 && first < 1.5 {
                let mut dst = usize::MAX;
                let mut dd = 1e18;
                let mut c = comp & !(1u64 << i) & !(v.hostile_inbound);
                while c != 0 {
                    let k = c.trailing_zeros() as usize;
                    c &= c - 1;
                    if w.map.d(i, k) < dd {
                        dd = w.map.d(i, k);
                        dst = k;
                    }
                }
                if dst != usize::MAX {
                    out.push(Action::Send { src: i as u8, dst: dst as u8, amount: amt });
                    used |= 1u64 << i;
                }
            }
        }
    }

    // 2. Attacks: each free border state takes its best worthwhile target.
    let mut b = v.border & !used;
    while b != 0 {
        let i = b.trailing_zeros() as usize;
        b &= b - 1;
        let have = s.troops[i].floor();
        if have < 2.0 {
            continue;
        }
        let over_cap = s.troops[i] >= s.cap(i);
        let mut best_j = usize::MAX;
        let mut best_net = if over_cap { p.theta - p.cap_relief } else { p.theta };
        let mut best_amt = 0.0;
        let mut t = w.map.adj_mask[i] & !v.mine & !claimed;
        while t != 0 {
            let j = t.trailing_zeros() as usize;
            t &= t - 1;
            let (po, pt, eta) = at_arrival(w, s, &v, i, j);
            if po == me {
                continue; // already being taken by troops in flight
            }
            if po != NEUTRAL && hoarding {
                continue;
            }
            // leave a guard against the strongest enemy neighbour other than the target
            let mut threat = 0.0f64;
            if v.contact >> i & 1 == 1 {
                let mut e = w.map.adj_mask[i] & !v.mine & !(1u64 << j);
                while e != 0 {
                    let k = e.trailing_zeros() as usize;
                    e &= e - 1;
                    if s.owner[k] != NEUTRAL {
                        threat = threat.max(s.troops[k]);
                    }
                }
            }
            let avail = have - (p.guard * threat).floor();
            let tm = s.def_mult(w, j);
            let need = need_to_take(pt, s.tile[j], tm);
            if avail < need * p.margin || avail < 1.0 {
                continue;
            }
            let dmg = if s.tile[j] == TILE_FORTRESS && pt > 0.0 { js_round(avail * FORTRESS_DEF) } else { avail };
            let surv = lanchester(dmg, pt, tm);
            if surv <= 0.0 {
                continue;
            }
            let net = capture_gain(w, s, p, me, j, po, pt) - (avail - surv) - p.eta_w * eta;
            if net > best_net {
                best_net = net;
                best_j = j;
                best_amt = avail;
            }
        }
        if best_j != usize::MAX {
            out.push(Action::Send { src: i as u8, dst: best_j as u8, amount: best_amt });
            used |= 1u64 << i;
            claimed |= 1u64 << best_j;
        }
    }

    // 3. Economy: safe states buy growth (and, optionally, cap).
    let mut e = v.mine & !used & !v.contact & !v.hostile_inbound;
    while e != 0 {
        let i = e.trailing_zeros() as usize;
        e &= e - 1;
        if w.rules.depth && s.inv[i] < p.invest_max && s.troops[i] >= invest_cost(s.inv[i]) + p.invest_reserve + 1.0 {
            out.push(Action::Invest(i as u8));
            used |= 1u64 << i;
        } else if p.upgrade && w.rules.upgrades && s.level[i] < MAX_LEVEL && s.troops[i] >= s.cap(i) - 1.0 && s.troops[i] >= upgrade_cost(s.level[i]) {
            out.push(Action::Upgrade(i as u8));
            used |= 1u64 << i;
        }
    }

    // 4. Logistics: interior stacks move up to the border state that needs them most.
    let mut pending = [0.0f64; MAXN]; // my troops already heading to each state
    if v.mine & !v.border & !used != 0 {
        for o in &s.orbs {
            if o.owner == me {
                pending[o.dst as usize] += o.count;
            }
        }
    }
    let mut inner = v.mine & !v.border & !used;
    while inner != 0 {
        let i = inner.trailing_zeros() as usize;
        inner &= inner - 1;
        let amt = s.troops[i].floor();
        if amt < p.forward_min {
            continue;
        }
        if s.inv[i] < p.invest_max && amt < invest_cost(s.inv[i]) + p.invest_reserve + 1.0 {
            continue; // saving up for the next invest tier
        }
        let comp = State::flood(w, v.mine, i);
        let mut best = usize::MAX;
        let mut best_score = -1e18;
        let mut c = comp & v.border;
        while c != 0 {
            let k = c.trailing_zeros() as usize;
            c &= c - 1;
            // do not pile troops above the cap: the excess just bleeds away
            if s.troops[k] + pending[k] + amt > s.cap(k) + p.cap_slack {
                continue;
            }
            let pressure = if v.contact >> k & 1 == 1 { 0.6 + (v.enemy_adj_max[k] - s.troops[k]).max(0.0) / 100.0 } else { 0.0 };
            let score = pressure - w.map.d(i, k) / w.map.diag;
            if score > best_score {
                best_score = score;
                best = k;
            }
        }
        if best != usize::MAX {
            out.push(Action::Send { src: i as u8, dst: best as u8, amount: amt });
            pending[best] += amt;
        }
    }
    let _ = n;
}

/// The tunable numbers, in a fixed order, with their search ranges (used by the tuner and the
/// text format).
pub const FIELDS: [(&str, f64, f64); 12] = [
    ("margin", 1.0, 2.5),
    ("theta", -15.0, 25.0),
    ("kg", 5.0, 150.0),
    ("enemy_bias", 0.2, 4.0),
    ("neutral_bias", 0.2, 4.0),
    ("guard", 0.0, 1.5),
    ("invest_max", 0.0, 3.49),
    ("invest_reserve", 0.0, 40.0),
    ("forward_min", 2.0, 40.0),
    ("eta_w", 0.0, 8.0),
    ("tile_bonus", 0.0, 50.0),
    ("cap_slack", 0.0, 100.0),
];

impl PolicyParams {
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.margin, self.theta, self.kg, self.enemy_bias, self.neutral_bias, self.guard, self.invest_max as f64, self.invest_reserve, self.forward_min, self.eta_w, self.tile_bonus, self.cap_slack]
    }

    /// Build from a vector in `FIELDS` order (values are clamped to their ranges). Fields that are
    /// not in the vector keep the values of `self`.
    pub fn with_vec(&self, v: &[f64]) -> PolicyParams {
        let c = |k: usize| v[k].clamp(FIELDS[k].1, FIELDS[k].2);
        PolicyParams {
            margin: c(0),
            theta: c(1),
            kg: c(2),
            enemy_bias: c(3),
            neutral_bias: c(4),
            guard: c(5),
            invest_max: c(6).floor() as u8,
            invest_reserve: c(7),
            forward_min: c(8),
            eta_w: c(9),
            tile_bonus: c(10),
            cap_slack: c(11),
            ..self.clone()
        }
    }

    /// One line of `key=value` pairs.
    pub fn to_text(&self) -> String {
        let v = self.to_vec();
        let mut s: Vec<String> = FIELDS.iter().zip(v.iter()).map(|(f, x)| format!("{}={:.4}", f.0, x)).collect();
        s.push(format!("upgrade={}", self.upgrade as u8));
        s.push(format!("hoard_ratio={:.3}", self.hoard_ratio));
        s.push(format!("defend={}", self.defend as u8));
        s.push(format!("cap_relief={:.3}", self.cap_relief));
        s.join(" ")
    }

    pub fn from_text(text: &str) -> PolicyParams {
        let mut p = PolicyParams::default();
        let mut v = p.to_vec();
        for kv in text.split_whitespace() {
            let (k, val) = kv.split_once('=').unwrap_or_else(|| panic!("bad policy field {kv}"));
            let x: f64 = val.parse().unwrap_or_else(|_| panic!("bad policy value {kv}"));
            if let Some(idx) = FIELDS.iter().position(|f| f.0 == k) {
                v[idx] = x;
            } else {
                match k {
                    "upgrade" => p.upgrade = x != 0.0,
                    "hoard_ratio" => p.hoard_ratio = x,
                    "defend" => p.defend = x != 0.0,
                    "cap_relief" => p.cap_relief = x,
                    _ => panic!("unknown policy field {k}"),
                }
            }
        }
        p.with_vec(&v)
    }
}
