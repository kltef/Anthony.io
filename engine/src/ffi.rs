//! C ABI for the WebAssembly build. All data crosses in one shared f64 buffer, so the JavaScript
//! side needs no generated bindings: write numbers, call a function, read numbers back.
//!
//! Layouts (all f64):
//!   world  : n, depth, adjacency, terrain, upgrades, orb_speed, owners, grow[owners],
//!            then per state: cx, cy, degree, neighbour ids...
//!   state  : time, supply_clock, then per state: owner, troops, grow_acc, level, tile, inv, off,
//!            then orb count, then per orb: owner, src, dst, count, remaining, total
//!   actions: per action: kind (0 send, 1 invest, 2 upgrade), a, b, amount

use crate::map::Map;
use crate::policy::{self, PolicyParams};
use crate::rules::*;
use crate::search::{SearchParams, Searcher};
use std::cell::RefCell;

const BUF: usize = 16384;

enum Brain {
    Script(PolicyParams),
    Search(Box<Searcher>),
}

struct Ctx {
    buf: Vec<f64>,
    world: Option<World>,
    brains: Vec<Option<Brain>>,
    state: State,
    out: Vec<Action>,
}

thread_local! {
    static CTX: RefCell<Ctx> = RefCell::new(Ctx { buf: vec![0.0; BUF], world: None, brains: Vec::new(), state: State::empty(0), out: Vec::new() });
}

/// Address of the shared buffer (`sio_buf_len()` f64 values).
#[no_mangle]
pub extern "C" fn sio_buf() -> *mut f64 {
    CTX.with(|c| c.borrow_mut().buf.as_mut_ptr())
}

#[no_mangle]
pub extern "C" fn sio_buf_len() -> u32 {
    BUF as u32
}

/// Load the world description from the buffer. Returns the number of states (0 on bad input).
#[no_mangle]
pub extern "C" fn sio_world() -> u32 {
    CTX.with(|c| {
        let mut c = c.borrow_mut();
        let b = &c.buf;
        let n = b[0] as usize;
        if n == 0 || n > crate::map::MAXN {
            return 0;
        }
        let owners = (b[6] as usize).min(MAX_OWNERS);
        let mut grow = [0.0; MAX_OWNERS];
        grow[..owners].copy_from_slice(&b[7..7 + owners]);
        let mut k = 7 + owners;
        let (mut cx, mut cy, mut adj) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            cx.push(b[k]);
            cy.push(b[k + 1]);
            let deg = b[k + 2] as usize;
            adj.push(b[k + 3..k + 3 + deg].iter().map(|x| *x as u8).collect());
            k += 3 + deg;
        }
        let map = Map::new("live", vec![String::new(); n], cx, cy, adj);
        let rules = Rules { depth: b[1] != 0.0, adjacency: b[2] != 0.0, terrain: b[3] != 0.0, upgrades: b[4] != 0.0, grow, orb_speed: b[5] };
        c.world = Some(World::new(map, rules));
        c.brains.clear();
        n as u32
    })
}

/// Create the brain for `owner` from a spec held in the buffer as `len` character codes:
/// a policy name (`base`, `t4`, ...) or `search` / `search:key=value,...`. Returns 1 on success.
#[no_mangle]
pub extern "C" fn sio_agent(owner: u32, len: u32) -> u32 {
    CTX.with(|c| {
        let mut c = c.borrow_mut();
        let spec: String = c.buf[..len as usize].iter().map(|x| *x as u8 as char).collect();
        let brain = if spec == "search" || spec.starts_with("search:") {
            let mut p = SearchParams::default();
            if let Some((_, opts)) = spec.split_once(':') {
                for kv in opts.split(',') {
                    match kv.split_once('=') {
                        Some((k, v)) => p.set(k, v),
                        None => return 0,
                    }
                }
            }
            let mut s = Searcher::new(&spec, p);
            s.set_owner(owner as u8);
            Brain::Search(Box::new(s))
        } else {
            match crate::roster::policy(&spec) {
                Some(p) => Brain::Script(p),
                None => return 0,
            }
        };
        let o = owner as usize;
        if c.brains.len() <= o {
            c.brains.resize_with(o + 1, || None);
        }
        c.brains[o] = Some(brain);
        1
    })
}

/// Decide `owner`'s orders for the state in the buffer. Writes the actions back into the buffer
/// and returns how many there are.
#[no_mangle]
pub extern "C" fn sio_decide(owner: u32) -> u32 {
    CTX.with(|c| {
        let mut guard = c.borrow_mut();
        let c = &mut *guard;
        let w = match &c.world {
            Some(w) => w,
            None => return 0,
        };
        let n = w.map.n;
        let b = &c.buf;
        let st = &mut c.state;
        *st = State::empty(n);
        st.time = b[0];
        st.supply_clock = b[1];
        let mut k = 2;
        for i in 0..n {
            st.owner[i] = b[k] as u8;
            st.troops[i] = b[k + 1];
            st.grow_acc[i] = b[k + 2];
            st.level[i] = (b[k + 3] as u8).min(MAX_LEVEL);
            st.tile[i] = b[k + 4] as u8;
            st.inv[i] = (b[k + 5] as u8).min(MAX_INVEST);
            st.off[i] = b[k + 6] != 0.0;
            k += 7;
        }
        let orbs = b[k] as usize;
        k += 1;
        for u in 0..orbs {
            if k + 6 > BUF {
                break;
            }
            st.orbs.push(Orb { owner: b[k] as u8, src: b[k + 1] as u8, dst: b[k + 2] as u8, count: b[k + 3], rem: b[k + 4], total: b[k + 5], uid: u as u32 });
            k += 6;
        }
        st.next_uid = orbs as u32;
        c.out.clear();
        match c.brains.get_mut(owner as usize).and_then(|x| x.as_mut()) {
            Some(Brain::Script(p)) => policy::act(w, st, owner as u8, p, &mut c.out),
            Some(Brain::Search(s)) => s.decide(w, st, &mut c.out),
            None => return 0,
        }
        let mut k = 0;
        for a in &c.out {
            let (kind, x, y, amt) = match *a {
                Action::Send { src, dst, amount } => (0.0, src as f64, dst as f64, amount),
                Action::Invest(i) => (1.0, i as f64, 0.0, 0.0),
                Action::Upgrade(i) => (2.0, i as f64, 0.0, 0.0),
            };
            c.buf[k] = kind;
            c.buf[k + 1] = x;
            c.buf[k + 2] = y;
            c.buf[k + 3] = amt;
            k += 4;
        }
        c.out.len() as u32
    })
}

/// Rollouts used by `owner`'s last decision (0 for scripted brains). For the debug panel.
#[no_mangle]
pub extern "C" fn sio_last_rollouts(owner: u32) -> u32 {
    CTX.with(|c| match c.borrow().brains.get(owner as usize).and_then(|x| x.as_ref()) {
        Some(Brain::Search(s)) => s.last.rollouts as u32,
        _ => 0,
    })
}
