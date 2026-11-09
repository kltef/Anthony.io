//! Measure an agent's play style with the same statistics as the tester's exported games, so a
//! bot fitted to him can be checked against his real numbers.
//!   usage: profile <agent> <opponent> [--games N] [--seed S] [--map NAME]

use stateio::agent::{Agent, TICK};
use stateio::rules::*;
use stateio::setup::{new_game, Setup};
use stateio::{roster, Map, Rng};
use std::collections::HashMap;

fn q(v: &mut Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let (mut games, mut seed, mut map_name) = (40usize, 3u64, "Continental U.S.".to_string());
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => { games = args[i + 1].parse().unwrap(); i += 1; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 1; }
            "--map" => { map_name = args[i + 1].clone(); i += 1; }
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    let (spec, opp) = (pos[0].clone(), pos[1].clone());
    let map = Map::builtin().into_iter().find(|m| m.name == map_name).expect("unknown map");
    let w = World::new(map.clone(), Rules::standard(&map, 1.0));
    let (mut wins, mut moves, mut mins) = (0, 0usize, 0.0);
    let (mut k_re, mut k_ne, mut k_en) = (0usize, 0usize, 0usize);
    let (mut f_all, mut f_half) = (0usize, 0usize);
    let mut joint: HashMap<usize, usize> = HashMap::new();
    let (mut first_enemy, mut ratio_n, mut ratio_e, mut force_e) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let checkpoints = [30.0, 60.0, 90.0, 120.0, 150.0];
    let mut traj: Vec<Vec<[f64; 4]>> = vec![Vec::new(); checkpoints.len()];
    for g in 0..games {
        let gs = seed * 1000 + g as u64;
        let mut s = new_game(&w.map, &Setup::duel(), &mut Rng::new(gs));
        s.recompute_supply(&w);
        // alternate seats so map luck cancels
        let me: u8 = if g % 2 == 0 { 1 } else { 2 };
        let other = 3 - me;
        let mut a = roster::make(&spec);
        let mut b = roster::make(&opp);
        a.reset(&w, me, gs);
        b.reset(&w, other, gs + 1);
        let (mut ba, mut bb) = (Vec::new(), Vec::new());
        let mut fe: Option<f64> = None;
        let mut ci = 0;
        while s.winner().is_none() && s.time < 600.0 {
            ba.clear();
            bb.clear();
            a.act(&w, &s, &mut ba);
            b.act(&w, &s, &mut bb);
            // classify my sends like the game's move log (one entry per source state)
            let mut groups: HashMap<u8, (f64, f64, usize)> = HashMap::new();
            for act in &ba {
                if let Action::Send { src, dst, amount } = *act {
                    let (si, di) = (src as usize, dst as usize);
                    if s.owner[si] != me || s.troops[si] < 1.0 {
                        continue;
                    }
                    moves += 1;
                    let sent = amount.min(s.troops[si].floor());
                    if sent >= s.troops[si].floor() - 0.5 { f_all += 1 } else { f_half += 1 }
                    let e = groups.entry(dst).or_insert((0.0, s.troops[di], 0));
                    e.0 += sent;
                    e.2 += 1;
                    if s.owner[di] == me {
                        k_re += 1;
                    } else if s.owner[di] == NEUTRAL {
                        k_ne += 1;
                        ratio_n.push(sent / s.troops[di].max(1.0));
                    } else {
                        k_en += 1;
                        if fe.is_none() {
                            fe = Some(s.time);
                        }
                    }
                }
            }
            for (dst, (tot, def, n)) in groups {
                *joint.entry(n.min(5)).or_default() += n;
                let o = s.owner[dst as usize];
                if o != me && o != NEUTRAL {
                    force_e.push(tot);
                    ratio_e.push(tot / def.max(1.0));
                }
            }
            for &x in &ba {
                s.apply(&w, me, x);
            }
            for &x in &bb {
                s.apply(&w, other, x);
            }
            s.step(&w, TICK);
            while ci < checkpoints.len() && s.time >= checkpoints[ci] {
                let (tm, cm) = s.totals(me);
                let (to, co) = s.totals(other);
                traj[ci].push([cm as f64, tm, co as f64, to]);
                ci += 1;
            }
        }
        if s.winner() == Some(me) {
            wins += 1;
        }
        mins += s.time / 60.0;
        first_enemy.push(fe.unwrap_or(s.time));
    }
    let n = moves.max(1) as f64;
    println!("{spec} vs {opp}: {} games, wins {} ({:.0}%)", games, wins, 100.0 * wins as f64 / games as f64);
    println!("  moves/min {:.0}   own-to-own {:.0}%  neutral {:.0}%  enemy {:.0}%   send all {:.0}% half {:.0}%",
        moves as f64 / mins, 100.0 * k_re as f64 / n, 100.0 * k_ne as f64 / n, 100.0 * k_en as f64 / n, 100.0 * f_all as f64 / n, 100.0 * f_half as f64 / n);
    let mut js: Vec<_> = joint.into_iter().collect();
    js.sort();
    println!("  states per move group {}", js.iter().map(|(k, v)| format!("{}{}:{:.0}%", k, if *k == 5 { "+" } else { "" }, 100.0 * *v as f64 / n)).collect::<Vec<_>>().join(" "));
    println!("  first enemy attack median {:.0}s   neutral force/defender median {:.2}   enemy force median {:.0}, force/defender median {:.1}",
        q(&mut first_enemy, 0.5), q(&mut ratio_n, 0.5), q(&mut force_e, 0.5), q(&mut ratio_e, 0.5));
    for (ci, t) in checkpoints.iter().enumerate() {
        let r = &mut traj[ci];
        if r.len() < 3 {
            continue;
        }
        let mut col = |k: usize| { let mut v: Vec<f64> = r.iter().map(|x| x[k]).collect(); q(&mut v, 0.5) };
        println!("  t={:3.0} n={:3}  me {:3.0} st {:5.0} tr | opp {:3.0} st {:5.0} tr", t, r.len(), col(0), col(1), col(2), col(3));
    }
}
