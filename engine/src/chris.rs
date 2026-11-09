//! A bot that plays like the play-tester ("Chris"), fitted to his exported games
//! (phone_export/stateio_playdata.json, analysed by tools/chris_style.py).
//!
//! What his logs show against the engine AI (22 Nightmare / Nightmare+ games, 15 wins):
//!   * ~80% of his moves move troops between his own states, usually sweeping 5+ small states
//!     into one stack at a time (consolidation into hubs).
//!   * Neutrals are taken with just enough: force about 1.3x the defender.
//!   * The enemy is left alone for the first ~90 s, then hit only where his force is
//!     overwhelming (median 13x the defender), with all troops (64%) or half (35%), often from
//!     several states at once.
//!   * His troops pile up (hundreds by 2 minutes) while the AI's stay near zero.
//!
//! The bot reproduces those habits with a handful of parameters; `tools/chris_style.py` compares
//! the bot's measured habits with his.

use crate::map::MAXN;
use crate::policy::{at_arrival, View};
use crate::rules::*;

#[derive(Clone, Debug)]
pub struct ChrisParams {
    /// Seconds between decisions (he makes ~70 single-state moves a minute, mostly in sweeps).
    pub cadence: f64,
    /// Take a neutral when the force reaching it is at least this multiple of what it takes.
    pub neutral_margin: f64,
    /// Attack an enemy state only with at least this multiple of what it takes.
    pub enemy_ratio: f64,
    /// Smallest stack a feeder bothers to send to its hub.
    pub min_feed: f64,
    /// A hub stops receiving when it holds this much above its cap.
    pub hub_slack: f64,
    /// Combine several adjacent states into one attack (his drag-sweep joint attacks).
    pub joint: bool,
    /// When half of a stack already meets `enemy_ratio`, send half and keep the rest.
    pub send_half: bool,
    /// Do not attack the enemy before this many seconds (he waits ~60-100 s).
    pub enemy_after: f64,
}

impl Default for ChrisParams {
    fn default() -> Self {
        ChrisParams { cadence: 0.5, neutral_margin: 1.3, enemy_ratio: 6.0, min_feed: 2.0, hub_slack: 20.0, joint: true, send_half: true, enemy_after: 0.0 }
    }
}

impl ChrisParams {
    pub fn set(&mut self, key: &str, val: &str) -> Result<(), String> {
        let num = || val.parse::<f64>().map_err(|_| format!("bad number for {key}: {val}"));
        match key {
            "cadence" => self.cadence = num()?,
            "nm" | "neutral_margin" => self.neutral_margin = num()?,
            "er" | "enemy_ratio" => self.enemy_ratio = num()?,
            "feed" | "min_feed" => self.min_feed = num()?,
            "slack" | "hub_slack" => self.hub_slack = num()?,
            "joint" => self.joint = num()? != 0.0,
            "half" => self.send_half = num()? != 0.0,
            "after" => self.enemy_after = num()?,
            _ => return Err(format!("unknown chris option {key}")),
        }
        Ok(())
    }
}

/// Orders for this instant. At most one order per source state.
pub fn act(w: &World, s: &State, me: u8, p: &ChrisParams, out: &mut Vec<Action>) {
    let v = View::new(w, s, me);
    if v.mine == 0 {
        return;
    }
    let mut used = 0u64;

    // 1. Strikes: enemy states where the force I can land is overwhelming, then cheap neutrals.
    //    Cheapest targets first (he "pounces on the weakest").
    let mut targets: Vec<(f64, usize, bool)> = Vec::new();
    let mut t = 0u64;
    let mut m = v.mine;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        t |= w.map.adj_mask[i] & !v.mine;
    }
    while t != 0 {
        let j = t.trailing_zeros() as usize;
        t &= t - 1;
        let enemy = s.owner[j] != NEUTRAL;
        if enemy && s.time < p.enemy_after {
            continue;
        }
        targets.push((s.troops[j], j, enemy));
    }
    targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (_, j, enemy) in targets {
        // my free states next to the target, biggest first
        let mut srcs: Vec<(f64, usize)> = Vec::new();
        let mut a = v.mine & w.map.adj_mask[j] & !used;
        while a != 0 {
            let i = a.trailing_zeros() as usize;
            a &= a - 1;
            if s.troops[i] >= 2.0 {
                srcs.push((s.troops[i].floor(), i));
            }
        }
        if srcs.is_empty() {
            continue;
        }
        srcs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        // what it will take when the biggest stack's orb arrives
        let (po, pt, _) = at_arrival(w, s, &v, srcs[0].1, j);
        if po == me {
            continue;
        }
        let need = need_to_take(pt, s.tile[j], s.def_mult(w, j));
        let want = if enemy { p.enemy_ratio * need.max(1.0) } else { p.neutral_margin * need };
        if !p.joint {
            srcs.truncate(1);
        }
        // smallest prefix of sources (biggest first) that reaches `want`
        let mut sum = 0.0;
        let mut k = 0;
        while k < srcs.len() && sum < want {
            sum += srcs[k].0;
            k += 1;
        }
        if sum < want {
            continue;
        }
        let half = p.send_half && enemy && k == 1 && srcs[0].0 * 0.5 >= want;
        for &(have, i) in &srcs[..k] {
            out.push(Action::Send { src: i as u8, dst: j as u8, amount: if half { (have * 0.5).floor() } else { have } });
            used |= 1u64 << i;
        }
    }

    // 2. Consolidation: every other stack in a connected territory sweeps into its hubs. Hubs are
    //    the biggest border states (fronts facing the enemy first), so the sweep builds a few big
    //    stacks where they can strike.
    let mut pending = [0.0f64; MAXN];
    for o in &s.orbs {
        if o.owner == me {
            pending[o.dst as usize] += o.count;
        }
    }
    let mut seen = 0u64;
    let mut m = v.mine;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        if seen >> i & 1 == 1 {
            continue;
        }
        let comp = State::flood(w, v.mine, i);
        seen |= comp;
        // hub candidates: contact states if any, else border states
        let front = if comp & v.contact != 0 { comp & v.contact } else { comp & v.border };
        if front == 0 {
            continue;
        }
        let mut hubs: Vec<usize> = Vec::new();
        let mut f = front;
        while f != 0 {
            let k = f.trailing_zeros() as usize;
            f &= f - 1;
            hubs.push(k);
        }
        hubs.sort_by(|&a, &b| s.troops[b].partial_cmp(&s.troops[a]).unwrap());
        hubs.truncate(2);
        let mut c = comp & !used;
        while c != 0 {
            let k = c.trailing_zeros() as usize;
            c &= c - 1;
            if hubs.contains(&k) {
                continue;
            }
            let have = s.troops[k].floor();
            if have < p.min_feed {
                continue;
            }
            // nearest hub that is not overfull
            let mut best = usize::MAX;
            let mut bd = f64::MAX;
            for &h in &hubs {
                if s.troops[h] + pending[h] + have > s.cap(h) + p.hub_slack {
                    continue;
                }
                if w.map.d(k, h) < bd {
                    bd = w.map.d(k, h);
                    best = h;
                }
            }
            if best != usize::MAX {
                out.push(Action::Send { src: k as u8, dst: best as u8, amount: have });
                pending[best] += have;
                used |= 1u64 << k;
            }
        }
    }
}

pub struct Chris {
    pub label: String,
    pub p: ChrisParams,
    me: u8,
    next: f64,
}

impl Chris {
    pub fn new(label: &str, p: ChrisParams) -> Chris {
        Chris { label: label.to_string(), p, me: 0, next: 0.0 }
    }
}

impl crate::agent::Agent for Chris {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn reset(&mut self, _w: &World, me: u8, seed: u64) {
        self.me = me;
        self.next = 0.2 + (seed % 7) as f64 * 0.03;
    }
    fn act(&mut self, w: &World, s: &State, out: &mut Vec<Action>) {
        if s.time < self.next {
            return;
        }
        self.next = s.time + self.p.cadence;
        act(w, s, self.me, &self.p, out);
    }
}
