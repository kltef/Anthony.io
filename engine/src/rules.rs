//! The State.io rules. This is the single source of truth: the game, the search and the training
//! all step this code. The behaviour is specified in `RULES.md` at the repo root and checked
//! against the original JavaScript simulation by `tests/conformance`.

use crate::map::{Map, MAXN};

pub const NEUTRAL: u8 = 0;
/// Owner ids run 0 (neutral), 1 (first human), 2.. (other teams). Scenario mode goes up to 64.
pub const MAX_OWNERS: usize = 65;

/// Cap by upgrade level. Every non-neutral owner (human or AI) uses the same tiers.
pub const CAP_TIERS: [f64; 5] = [150.0, 175.0, 200.0, 225.0, 250.0];
pub const MAX_LEVEL: u8 = 4;
pub const MAX_INVEST: u8 = 3;

pub const TILE_NONE: u8 = 0;
pub const TILE_CAPITAL: u8 = 1;
pub const TILE_FACTORY: u8 = 2;
pub const TILE_FORTRESS: u8 = 3;

/// Lanchester concentration exponent.
pub const LANCH: f64 = 1.5;
/// Over-cap attrition: loss per second = ATTRITION_Q * excess^2 / cap.
pub const ATTRITION_Q: f64 = 0.06;
/// Off-supply states defend at this multiplier and grow at `OFFSUP_GROW`.
pub const OFFSUP_DEF: f64 = 0.85;
pub const OFFSUP_GROW: f64 = 0.4;
/// Fortress tiles absorb 25% of an incoming attack.
pub const FORTRESS_DEF: f64 = 0.75;
/// Neutral states restock at this rate up to `RESTOCK_MAX` (depth rules).
pub const RESTOCK_RATE: f64 = 0.2;
pub const RESTOCK_MAX: f64 = 30.0;
/// Supply (connected-core) recompute period in seconds.
pub const SUPPLY_PERIOD: f64 = 0.5;
/// An orb lands when it is within one step plus this many pixels of its target.
pub const ARRIVE_SLOP: f64 = 2.0;

#[inline]
pub fn upgrade_cost(level: u8) -> f64 {
    40.0 + level as f64 * 30.0
}
#[inline]
pub fn invest_cost(inv: u8) -> f64 {
    25.0 + inv as f64 * 15.0
}
#[inline]
pub fn tile_grow(tile: u8) -> f64 {
    match tile {
        TILE_CAPITAL => 1.35,
        TILE_FACTORY => 1.6,
        _ => 1.0,
    }
}
#[inline]
pub fn tile_cap_bonus(tile: u8) -> f64 {
    match tile {
        TILE_CAPITAL | TILE_FORTRESS => 40.0,
        _ => 0.0,
    }
}
/// JavaScript `Math.round` (round half up).
#[inline]
pub fn js_round(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// Rule switches and per-owner economy. Everything the simulation needs besides the map.
#[derive(Clone, Debug)]
pub struct Rules {
    /// Depth mechanics: Lanchester combat, supply, invest, attrition, neutral restock.
    pub depth: bool,
    /// Risk-style movement (attack adjacent, reinforce through owned chain). Off = send anywhere.
    pub adjacency: bool,
    /// Terrain defence bonus for low-degree states (1.4x for <=2 neighbours, 1.2x for 3).
    pub terrain: bool,
    /// Cap upgrades available.
    pub upgrades: bool,
    /// Troop growth per second, per owner (0 for neutral).
    pub grow: [f64; MAX_OWNERS],
    /// Orb travel speed in map units per second.
    pub orb_speed: f64,
}

impl Rules {
    /// The standard solo/multiplayer rule set with every owner growing at `grow`.
    pub fn standard(map: &Map, grow: f64) -> Rules {
        let mut g = [grow; MAX_OWNERS];
        g[NEUTRAL as usize] = 0.0;
        Rules { depth: true, adjacency: true, terrain: true, upgrades: true, grow: g, orb_speed: map.diag / 8.0 }
    }
}

/// Map + rules + values derived from both. Shared (read-only) by every state in a search.
#[derive(Clone, Debug)]
pub struct World {
    pub map: Map,
    pub rules: Rules,
    /// Per-state terrain defence multiplier (1.0 when terrain is off).
    pub terr: Vec<f64>,
}

impl World {
    pub fn new(map: Map, rules: Rules) -> World {
        let terr = (0..map.n)
            .map(|i| {
                if !rules.terrain {
                    1.0
                } else {
                    match map.adj[i].len() {
                        0..=2 => 1.4,
                        3 => 1.2,
                        _ => 1.0,
                    }
                }
            })
            .collect();
        World { map, rules, terr }
    }
}

/// A group of troops in flight. Orbs travel in a straight line and cannot be recalled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orb {
    pub owner: u8,
    pub src: u8,
    pub dst: u8,
    pub count: f64,
    /// Remaining distance to the target.
    pub rem: f64,
    /// Total distance of the trip (for progress / rendering).
    pub total: f64,
    pub uid: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    /// Send `amount` troops (floored, at least 1) from `src` to `dst`.
    Send { src: u8, dst: u8, amount: f64 },
    /// Spend troops to raise a state's growth tier.
    Invest(u8),
    /// Spend troops to raise a state's cap tier.
    Upgrade(u8),
}

/// The full dynamic game state. Cheap to clone (fixed arrays + one orb vector).
#[derive(Clone, Debug)]
pub struct State {
    pub n: usize,
    pub owner: [u8; MAXN],
    pub troops: [f64; MAXN],
    pub grow_acc: [f64; MAXN],
    pub level: [u8; MAXN],
    pub tile: [u8; MAXN],
    pub inv: [u8; MAXN],
    pub off: [bool; MAXN],
    pub orbs: Vec<Orb>,
    pub time: f64,
    pub supply_clock: f64,
    pub next_uid: u32,
}

impl State {
    pub fn empty(n: usize) -> State {
        State {
            n,
            owner: [NEUTRAL; MAXN],
            troops: [0.0; MAXN],
            grow_acc: [0.0; MAXN],
            level: [0; MAXN],
            tile: [TILE_NONE; MAXN],
            inv: [0; MAXN],
            off: [false; MAXN],
            orbs: Vec::new(),
            time: 0.0,
            supply_clock: 0.0,
            next_uid: 0,
        }
    }

    /// Copy `src` into `self`, reusing the orb buffer (no allocation in the search hot path).
    #[inline]
    pub fn copy_from(&mut self, src: &State) {
        self.n = src.n;
        self.owner = src.owner;
        self.troops = src.troops;
        self.grow_acc = src.grow_acc;
        self.level = src.level;
        self.tile = src.tile;
        self.inv = src.inv;
        self.off = src.off;
        self.orbs.clear();
        self.orbs.extend_from_slice(&src.orbs);
        self.time = src.time;
        self.supply_clock = src.supply_clock;
        self.next_uid = src.next_uid;
    }

    #[inline]
    pub fn cap(&self, i: usize) -> f64 {
        let base = if self.owner[i] == NEUTRAL { CAP_TIERS[0] } else { CAP_TIERS[self.level[i] as usize] };
        base + tile_cap_bonus(self.tile[i])
    }

    /// Growth per second of state `i` under the current rules (0 for neutral or at/over cap).
    #[inline]
    pub fn growth(&self, w: &World, i: usize) -> f64 {
        let o = self.owner[i];
        if o == NEUTRAL {
            return 0.0;
        }
        let mut g = w.rules.grow[o as usize] * tile_grow(self.tile[i]);
        if w.rules.depth {
            g *= 1.0 + 0.4 * self.inv[i] as f64;
            if self.off[i] {
                g *= OFFSUP_GROW;
            }
        }
        g
    }

    /// Bitmask of states owned by `owner`.
    #[inline]
    pub fn owned_mask(&self, owner: u8) -> u64 {
        let mut m = 0u64;
        for i in 0..self.n {
            if self.owner[i] == owner {
                m |= 1u64 << i;
            }
        }
        m
    }

    /// Connected component (through `own` states) containing `start`, as a bitmask.
    #[inline]
    pub fn flood(w: &World, own: u64, start: usize) -> u64 {
        let mut comp = 1u64 << start;
        let mut frontier = comp;
        while frontier != 0 {
            let mut nb = 0u64;
            let mut f = frontier;
            while f != 0 {
                let i = f.trailing_zeros() as usize;
                f &= f - 1;
                nb |= w.map.adj_mask[i];
            }
            frontier = nb & own & !comp;
            comp |= frontier;
        }
        comp
    }

    /// Is `src -> dst` a legal send for `owner` right now?
    pub fn legal(&self, w: &World, owner: u8, src: usize, dst: usize) -> bool {
        if src == dst || src >= self.n || dst >= self.n || owner == NEUTRAL || self.owner[src] != owner {
            return false;
        }
        if !w.rules.adjacency {
            return true;
        }
        if self.owner[dst] != owner {
            return w.map.adjacent(src, dst);
        }
        State::flood(w, self.owned_mask(owner), src) >> dst & 1 == 1
    }

    /// Send troops. Returns false (and changes nothing) if the move is illegal or under 1 troop.
    pub fn send(&mut self, w: &World, owner: u8, src: usize, dst: usize, amount: f64) -> bool {
        if !self.legal(w, owner, src, dst) {
            return false;
        }
        let amount = amount.floor().min(self.troops[src].floor());
        if amount < 1.0 {
            return false;
        }
        self.troops[src] -= amount;
        let d = w.map.d(src, dst);
        self.next_uid += 1;
        self.orbs.push(Orb { owner, src: src as u8, dst: dst as u8, count: amount, rem: d, total: d, uid: self.next_uid });
        true
    }

    pub fn can_invest(&self, w: &World, owner: u8, i: usize) -> bool {
        w.rules.depth && i < self.n && owner != NEUTRAL && self.owner[i] == owner && self.inv[i] < MAX_INVEST
            && self.troops[i] > invest_cost(self.inv[i])
    }

    pub fn invest(&mut self, w: &World, owner: u8, i: usize) -> bool {
        if !self.can_invest(w, owner, i) {
            return false;
        }
        self.troops[i] -= invest_cost(self.inv[i]);
        self.inv[i] += 1;
        true
    }

    pub fn can_upgrade(&self, w: &World, owner: u8, i: usize) -> bool {
        w.rules.upgrades && i < self.n && owner != NEUTRAL && self.owner[i] == owner && self.level[i] < MAX_LEVEL
            && self.troops[i] >= upgrade_cost(self.level[i])
    }

    pub fn upgrade(&mut self, w: &World, owner: u8, i: usize) -> bool {
        if !self.can_upgrade(w, owner, i) {
            return false;
        }
        self.troops[i] -= upgrade_cost(self.level[i]);
        self.level[i] += 1;
        true
    }

    pub fn apply(&mut self, w: &World, owner: u8, a: Action) -> bool {
        match a {
            Action::Send { src, dst, amount } => self.send(w, owner, src as usize, dst as usize, amount),
            Action::Invest(i) => self.invest(w, owner, i as usize),
            Action::Upgrade(i) => self.upgrade(w, owner, i as usize),
        }
    }

    /// Effective defence multiplier of state `i` (terrain x supply). 1.0 without depth rules.
    #[inline]
    pub fn def_mult(&self, w: &World, i: usize) -> f64 {
        if !w.rules.depth {
            return 1.0;
        }
        w.terr[i] * if self.off[i] { OFFSUP_DEF } else { 1.0 }
    }

    /// Resolve `count` troops of `owner` landing on state `i`.
    #[inline]
    pub fn land(&mut self, w: &World, owner: u8, i: usize, count: f64) {
        let (o, t) = land_calc(w.rules.depth, self.owner[i], self.troops[i], self.tile[i], self.def_mult(w, i), owner, count);
        self.owner[i] = o;
        self.troops[i] = t;
    }

    /// Where state `i` will stand `horizon` seconds from now if nobody issues another order:
    /// growth plus every orb already in flight toward it, resolved in arrival order.
    /// Returns (owner, troops).
    pub fn project(&self, w: &World, i: usize, horizon: f64) -> (u8, f64) {
        self.project_with(w, i, horizon, None)
    }

    /// Like `project`, plus one hypothetical orb `(owner, count, eta_seconds)` heading to `i`.
    pub fn project_with(&self, w: &World, i: usize, horizon: f64, extra: Option<(u8, f64, f64)>) -> (u8, f64) {
        let speed = w.rules.orb_speed;
        let (mut owner, mut troops) = (self.owner[i], self.troops[i]);
        let tm = self.def_mult(w, i);
        let mut now = 0.0;
        let mut extra_done = false;
        let mut done = 0u64; // orbs already landed (bit per index, first 64 orbs to this state)
        let mut idx = [0usize; 64];
        let mut cnt = 0;
        for (k, o) in self.orbs.iter().enumerate() {
            if o.dst as usize == i && cnt < 64 {
                idx[cnt] = k;
                cnt += 1;
            }
        }
        loop {
            let mut best = usize::MAX;
            let mut best_eta = horizon;
            for c in 0..cnt {
                if done >> c & 1 == 0 {
                    let eta = ((self.orbs[idx[c]].rem - ARRIVE_SLOP).max(0.0)) / speed;
                    if eta <= best_eta {
                        best_eta = eta;
                        best = c;
                    }
                }
            }
            // the hypothetical orb lands in its place in the arrival order
            let mut use_extra = false;
            if let Some((_, _, eta)) = extra {
                if !extra_done && eta <= horizon && (best == usize::MAX || eta < best_eta) {
                    use_extra = true;
                    best_eta = eta;
                }
            }
            let until = if best == usize::MAX && !use_extra { horizon } else { best_eta };
            if owner != NEUTRAL {
                let cap = CAP_TIERS[self.level[i] as usize] + tile_cap_bonus(self.tile[i]);
                if troops < cap {
                    let mut g = w.rules.grow[owner as usize] * tile_grow(self.tile[i]);
                    if w.rules.depth {
                        g *= 1.0 + 0.4 * self.inv[i] as f64;
                    }
                    troops = (troops + g * (until - now)).min(cap.max(troops));
                }
            }
            now = until;
            if use_extra {
                extra_done = true;
                let (eo, ec, _) = extra.unwrap();
                let (no, nt) = land_calc(w.rules.depth, owner, troops, self.tile[i], tm, eo, ec);
                owner = no;
                troops = nt;
                continue;
            }
            if best == usize::MAX {
                break;
            }
            done |= 1u64 << best;
            let o = &self.orbs[idx[best]];
            let (no, nt) = land_calc(w.rules.depth, owner, troops, self.tile[i], tm, o.owner, o.count);
            owner = no;
            troops = nt;
        }
        (owner, troops)
    }

    /// Recompute which states are cut off from their owner's largest connected territory.
    pub fn recompute_supply(&mut self, w: &World) {
        self.off = [false; MAXN];
        let mut seen_owner = [false; MAX_OWNERS];
        for s in 0..self.n {
            let o = self.owner[s];
            if o == NEUTRAL || seen_owner[o as usize] {
                continue;
            }
            seen_owner[o as usize] = true;
            let own = self.owned_mask(o);
            // components in order of their lowest state id; the first of the largest size is the core
            let mut rest = own;
            let mut core = 0u64;
            let mut core_size = 0;
            while rest != 0 {
                let start = rest.trailing_zeros() as usize;
                let comp = State::flood(w, own, start);
                rest &= !comp;
                if comp.count_ones() > core_size {
                    core_size = comp.count_ones();
                    core = comp;
                }
            }
            let mut cut = own & !core;
            while cut != 0 {
                let i = cut.trailing_zeros() as usize;
                cut &= cut - 1;
                self.off[i] = true;
            }
        }
    }

    /// Advance the simulation by `dt` seconds: supply, growth/attrition/restock, orb travel, combat.
    pub fn step(&mut self, w: &World, dt: f64) {
        self.time += dt;
        let depth = w.rules.depth;
        if depth {
            self.supply_clock += dt;
            if self.supply_clock >= SUPPLY_PERIOD {
                self.supply_clock = 0.0;
                self.recompute_supply(w);
            }
        }
        for i in 0..self.n {
            let o = self.owner[i];
            let cap = self.cap(i);
            if o != NEUTRAL && self.troops[i] < cap {
                self.grow_acc[i] += self.growth(w, i) * dt;
                while self.grow_acc[i] >= 1.0 {
                    self.grow_acc[i] -= 1.0;
                    if self.troops[i] < cap {
                        self.troops[i] += 1.0;
                    }
                }
            } else if depth && o != NEUTRAL && self.troops[i] > cap {
                let ex = self.troops[i] - cap;
                self.troops[i] = cap.max(self.troops[i] - (ATTRITION_Q * ex * ex / cap) * dt);
            }
            if depth && o == NEUTRAL && self.troops[i] < RESTOCK_MAX {
                self.grow_acc[i] += RESTOCK_RATE * dt;
                while self.grow_acc[i] >= 1.0 {
                    self.grow_acc[i] -= 1.0;
                    if self.troops[i] < RESTOCK_MAX {
                        self.troops[i] += 1.0;
                    }
                }
            }
        }
        if !self.orbs.is_empty() {
            let step = w.rules.orb_speed * dt;
            // newest first, like the game: later sends resolve before earlier ones in the same step
            let mut k = self.orbs.len();
            while k > 0 {
                k -= 1;
                if self.orbs[k].rem <= step + ARRIVE_SLOP {
                    let o = self.orbs.remove(k);
                    self.land(w, o.owner, o.dst as usize, o.count);
                } else {
                    self.orbs[k].rem -= step;
                }
            }
        }
    }

    /// Bitmask (bit = owner id) of owners that still hold a state or have troops in flight.
    pub fn alive(&self) -> u128 {
        let mut m = 0u128;
        for i in 0..self.n {
            if self.owner[i] != NEUTRAL {
                m |= 1u128 << self.owner[i];
            }
        }
        for o in &self.orbs {
            m |= 1u128 << o.owner;
        }
        m
    }

    /// The sole surviving owner, if the game is over.
    pub fn winner(&self) -> Option<u8> {
        let m = self.alive();
        if m.count_ones() == 1 {
            Some(m.trailing_zeros() as u8)
        } else {
            None
        }
    }

    /// Total troops (on states and in flight) and state count for `owner`.
    pub fn totals(&self, owner: u8) -> (f64, usize) {
        let (mut t, mut c) = (0.0, 0);
        for i in 0..self.n {
            if self.owner[i] == owner {
                t += self.troops[i];
                c += 1;
            }
        }
        for o in &self.orbs {
            if o.owner == owner {
                t += o.count;
            }
        }
        (t, c)
    }
}

/// Troops left on the target after `atk` attackers hit `def` defenders with multiplier `tm`
/// (depth rules). Positive = attacker holds it with that many; negative = defender keeps that many.
#[inline]
pub fn lanchester(atk: f64, def: f64, tm: f64) -> f64 {
    let d = def * tm;
    if atk > d {
        js_round((atk.powf(LANCH) - d.powf(LANCH)).powf(1.0 / LANCH))
    } else {
        -js_round((d.powf(LANCH) - atk.powf(LANCH)).powf(1.0 / LANCH) / tm)
    }
}

/// Pure combat resolution: `count` troops of `atk_owner` land on a state held by `owner` with
/// `troops`, tile `tile` and defence multiplier `tm`. Returns the new (owner, troops).
#[inline]
pub fn land_calc(depth: bool, owner: u8, troops: f64, tile: u8, tm: f64, atk_owner: u8, count: f64) -> (u8, f64) {
    if owner == atk_owner {
        return (owner, troops + count);
    }
    let dmg = if tile == TILE_FORTRESS && troops > 0.0 { js_round(count * FORTRESS_DEF) } else { count };
    if depth {
        let def_eff = troops * tm;
        if dmg > def_eff {
            (atk_owner, js_round((dmg.powf(LANCH) - def_eff.powf(LANCH)).powf(1.0 / LANCH)))
        } else {
            (owner, js_round((def_eff.powf(LANCH) - dmg.powf(LANCH)).powf(1.0 / LANCH) / tm))
        }
    } else {
        let t = troops - dmg;
        if t < 0.0 {
            (atk_owner, -t)
        } else {
            (owner, t)
        }
    }
}

/// Smallest whole number of troops that captures a state defended by `troops` (multiplier `tm`).
#[inline]
pub fn need_to_take(troops: f64, tile: u8, tm: f64) -> f64 {
    let d = troops * tm;
    let n = if tile == TILE_FORTRESS && troops > 0.0 { d / FORTRESS_DEF } else { d };
    n.floor() + 1.0
}
