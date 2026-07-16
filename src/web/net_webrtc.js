// State.io — serverless P2P transport (WebRTC via PeerJS).
//
//   No game server required. This file reproduces the EXACT relay that
//   server.js provides, but peer-to-peer:
//
//     server.js model                         this file (P2P)
//     ------------------------------------    --------------------------------
//     POST /api/create  -> {room, token}      host creates a PeerJS peer whose
//                                             id IS the room code; returns it
//     GET  /api/stream  (SSE: events)         host<->guest WebRTC DataChannel
//     POST /api/host    (broadcast event)     host fans out over data channels
//     POST /api/state   (host snapshot)       host fans out 'state'
//     POST /api/action  (guest intent)        guest sends 'action' to host
//
//   How the connection is made WITHOUT a server of ours ("free signaling"):
//   WebRTC needs the two devices to swap a small handshake (SDP + ICE) before
//   they can talk directly. PeerJS does that swap for us over its FREE public
//   broker (0.peerjs.com) and uses Google's FREE public STUN servers to punch
//   through NAT. The broker only sees the one-time handshake — every snapshot
//   and intent after that travels DIRECTLY phone-to-phone, never touching any
//   server. The host stays authoritative exactly as before.
//
//   The game (index.html) calls P2P.api() and P2P.openStream() instead of
//   fetch()/EventSource when NET.p2p is true. Nothing in the game logic
//   changes — same events, same data shapes.

(function () {
  'use strict';

  // Same alphabet/length as server.js code4() so codes look identical and the
  // join screen (maxlength=4, uppercase) accepts them unchanged.
  var A36 = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  function code4() { var s = ''; for (var i = 0; i < 4; i++) s += A36[Math.floor(Math.random() * A36.length)]; return s; }
  function token() { return Math.random().toString(36).slice(2, 10); }

  // PeerJS ids share a global namespace on the broker, so prefix the 4-char
  // room code to avoid colliding with unrelated PeerJS apps.
  var NS = 'stateio-v1-';
  function peerId(room) { return NS + room; }

  // Free public STUN servers (no TURN — see README; symmetric-NAT pairs may
  // fail to connect, which is the one limitation of being fully serverless).
  var PEER_OPTS = {
    config: {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
      ]
    }
  };

  // ---- shared state ----
  var S = {
    role: null,        // 'host' | 'player'
    room: null,
    peer: null,        // PeerJS Peer
    dispatch: null,    // the game's event handler (set in openStream)
    // host only:
    conns: null,       // Map cid -> { conn, name, owner, cid }
    ownerNext: 2,      // owners 2..8, mirroring the server
    // guest only:
    conn: null         // DataConnection to the host
  };

  function havePeerJS() { return typeof window.Peer === 'function'; }

  // ---- host: fan a message out to every connected guest ----
  function broadcast(event, data) {
    if (!S.conns) return;
    S.conns.forEach(function (c) {
      try { if (c.conn && c.conn.open) c.conn.send({ t: event, data: data }); } catch (e) {}
    });
  }

  function rosterData() {
    var players = [];
    if (S.conns) S.conns.forEach(function (c) { players.push({ cid: c.cid, name: c.name, owner: c.owner }); });
    return { players: players };
  }

  function pushRoster() {
    var d = rosterData();
    broadcast('roster', d);
    if (S.dispatch) S.dispatch('roster', d);   // update the host's own lobby UI
  }

  // ---- host: a guest's data connection opened ----
  function onGuestConn(conn) {
    conn.on('open', function () {
      conn.on('data', function (msg) {
        if (!msg || !msg.t) return;
        if (msg.t === 'join') {
          var cid = msg.cid || token();
          var owner = Math.min(8, S.ownerNext++);
          S.conns.set(cid, { conn: conn, name: (msg.name || 'Player').slice(0, 14), owner: owner, cid: cid });
          conn._cid = cid;
          try { conn.send({ t: 'hello', data: { cid: cid, owner: owner } }); } catch (e) {}
          pushRoster();
        } else if (msg.t === 'action') {
          // forward the guest's intent to the host's game (applyRemoteAction)
          if (S.dispatch) S.dispatch('action', msg.data);
        }
      });
    });
    conn.on('close', function () {
      if (conn._cid && S.conns) { S.conns.delete(conn._cid); pushRoster(); }
    });
    conn.on('error', function () {
      if (conn._cid && S.conns) { S.conns.delete(conn._cid); pushRoster(); }
    });
  }

  // ---- the api() replacement: the game POSTs "to the server"; we route P2P ----
  // Returns a fetch-like object with .json() so callers using
  //   const r = await api(...); const j = await r.json();
  // keep working unchanged.
  function fakeResp(data) { return { ok: true, json: function () { return Promise.resolve(data || {}); } }; }

  function api(path, body) {
    var p = String(path).split('?')[0];

    if (p === '/api/create') {
      if (!havePeerJS()) return Promise.reject(new Error('PeerJS not loaded'));
      var room = code4();
      S.role = 'host';
      S.room = room;
      S.conns = new Map();
      S.ownerNext = 2;
      S.peer = new window.Peer(peerId(room), PEER_OPTS);
      S.peer.on('connection', onGuestConn);
      S.peer.on('error', function (err) {
        // 'unavailable-id' means the room code is already live on the broker —
        // astronomically unlikely with a fresh random code, but surface it.
        if (S.dispatch) S.dispatch('fatal', { msg: 'P2P error: ' + (err && err.type ? err.type : 'connection failed') });
      });
      return Promise.resolve(fakeResp({ room: room, token: token() }));
    }

    if (p === '/api/host')   { broadcast(body.event, body.data); return Promise.resolve(fakeResp()); }
    if (p === '/api/state')  { broadcast('state', body);          return Promise.resolve(fakeResp()); }
    if (p === '/api/result') { broadcast('result', body && body.result ? body.result : body); return Promise.resolve(fakeResp()); }

    if (p === '/api/action') {
      // guest -> host intent
      try { if (S.conn && S.conn.open) S.conn.send({ t: 'action', data: body }); } catch (e) {}
      return Promise.resolve(fakeResp());
    }

    // Server-side Grandmaster MCTS: there's no server in P2P mode, so REJECT —
    // this is what makes plannerMoveServer() fall back to the in-browser MCTS
    // (plannerMove), so the host still runs the trained Grandmaster AI locally.
    if (p === '/api/plan') return Promise.reject(new Error('no server in P2P mode'));

    // Other server-only extras (leaderboard, result recording) have no peer to
    // answer them. Resolve benignly so the game's .then()/.catch() chains don't
    // break; these features simply no-op in serverless mode.
    return Promise.resolve(fakeResp({}));
  }

  // ---- the openStream() replacement ----
  function openStream(role, name, dispatch, ctx) {
    ctx = ctx || {};
    S.role = role;
    S.dispatch = dispatch;

    if (role === 'host') {
      // Peer was already created in api('/api/create'); the host's lobby is live.
      // Re-emit the current roster in case anyone joined during setup.
      pushRoster();
      return;
    }

    // guest — room/cid are passed in from the game (NET is closure-local there).
    if (!havePeerJS()) { dispatch('fatal', { msg: 'PeerJS not loaded' }); return; }
    var room = ctx.room || S.room;
    var cid  = ctx.cid  || token();
    S.room = room;

    S.peer = new window.Peer(PEER_OPTS);   // random guest id
    S.peer.on('open', function () {
      var conn = S.peer.connect(peerId(room), { reliable: true });
      S.conn = conn;
      conn.on('open', function () {
        conn.send({ t: 'join', cid: cid, name: name || 'Player' });
      });
      conn.on('data', function (msg) {
        if (!msg || !msg.t) return;
        // host -> guest events map 1:1 onto the SSE events the game expects
        dispatch(msg.t, msg.data);
      });
      conn.on('close', function () { dispatch('fatal', { msg: 'Host disconnected.' }); });
      conn.on('error', function () { dispatch('fatal', { msg: 'Could not reach the host. Check the code.' }); });
    });
    S.peer.on('error', function (err) {
      var t = err && err.type ? err.type : 'connection failed';
      var msg = (t === 'peer-unavailable')
        ? 'No host found for that code. Is the host still in the lobby?'
        : 'P2P error: ' + t;
      dispatch('fatal', { msg: msg });
    });
  }

  function close() {
    try { if (S.conn) S.conn.close(); } catch (e) {}
    try { if (S.peer) S.peer.destroy(); } catch (e) {}
    S.peer = null; S.conn = null; S.conns = null; S.dispatch = null; S.role = null;
  }

  window.P2P = { api: api, openStream: openStream, close: close, _state: S };
})();
