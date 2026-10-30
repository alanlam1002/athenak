#!/usr/bin/env python3
"""Exact special-relativistic Riemann problem, ideal gas, velocities normal to the
interface (Marti & Mueller 1994, Living Rev. Relativ. 6, 7 (2003) Sec. 4).
State = (rho, p, v). solve(L, R, gam) -> sampler(xi) giving (rho, p, v) at xi = x/t.
Shocks: Taub adiabat + junction conditions (MM eq. for v_b uses W_s v_a/j); rarefactions:
the Riemann invariant
atanh(v) +- (2/sqrt(gam-1)) atanh(cs/sqrt(gam-1)) along the isentrope.
Self-check (main): the jump conditions across each shock in the shock frame."""
import numpy as np
from scipy.optimize import brentq


def h_of(rho, p, g):
    return 1.0 + g/(g - 1.0)*p/rho


def cs_of(rho, p, g):
    return np.sqrt(g*p/(rho*h_of(rho, p, g)))


def rare_invariant(cs, g):
    a = np.sqrt(g - 1.0)
    return 2.0/a*np.arctanh(cs/a)


def rarefaction(state, pstar, g, sign):
    """sign = -1: left wave; +1: right wave. Returns (rho*, v*)."""
    rho, p, v = state
    K = p/rho**g
    rs = (pstar/K)**(1.0/g)
    cs0, cs1 = cs_of(rho, p, g), cs_of(rs, pstar, g)
    # left (sign -1): atanh v + I(cs) const; right: atanh v - I(cs) const
    phi = np.arctanh(v) - sign*(rare_invariant(cs0, g) - rare_invariant(cs1, g))
    return rs, np.tanh(phi)


def shock(state, pstar, g, sign):
    """Returns (rho*, v*, Vs)."""
    rho, p, v = state
    h = h_of(rho, p, g)
    W = 1.0/np.sqrt(1.0 - v*v)
    # Taub adiabat for h* (Marti & Mueller eq. 31)
    a = 1.0 + (g - 1.0)*(p - pstar)/(g*pstar)
    b = -(g - 1.0)*(p - pstar)/(g*pstar)
    c = h*(p - pstar)/rho - h*h
    hs = (-b + np.sqrt(b*b - 4*a*c))/(2*a)
    rs = g*pstar/((g - 1.0)*(hs - 1.0))
    j2 = -(pstar - p)/(hs/rs - h/rho)
    j = sign*np.sqrt(j2)
    Vs = (rho*rho*W*W*v + j*np.sqrt(j2 + rho*rho*W*W*(1 - v*v)))/(rho*rho*W*W + j2)
    Ws = 1.0/np.sqrt(1.0 - Vs*Vs)
    vs = (h*W*v + Ws*(pstar - p)/j)/(h*W + (pstar - p)*(Ws*v/j + 1.0/(rho*W)))
    return rs, vs, Vs


def star_v(state, pstar, g, sign):
    if pstar > state[1]:
        return shock(state, pstar, g, sign)[1]
    return rarefaction(state, pstar, g, sign)[1]


def solve(L, R, g):
    f = lambda ps: star_v(L, ps, g, -1) - star_v(R, ps, g, +1)
    lo, hi = 1e-12*min(L[1], R[1]), 1e3*max(L[1], R[1])
    ps = brentq(f, lo, hi, xtol=1e-15*max(L[1], R[1]), rtol=1e-15, maxiter=500)
    vs = star_v(L, ps, g, -1)
    waves = {}
    for side, st, sg in (('L', L, -1), ('R', R, +1)):
        if ps > st[1]:
            rs, _, Vs = shock(st, ps, g, sg)
            waves[side] = ('shock', rs, Vs)
        else:
            rs, _ = rarefaction(st, ps, g, sg)
            lam = lambda rho, p, v, s: (v + s*cs_of(rho, p, g))/(1 + s*v*cs_of(rho, p, g))
            head = lam(*st, sg)
            tail = lam(rs, ps, vs, sg)
            waves[side] = ('rare', rs, head, tail)

    def fan(st, sg, xi):
        rho0, p0, v0 = st
        K = p0/rho0**g

        def g_of(rho):
            p = K*rho**g
            _, v = rarefaction(st, p, g, sg)
            c = cs_of(rho, p, g)
            return (v + sg*c)/(1 + sg*v*c) - xi
        rlo = waves['L' if sg < 0 else 'R'][1]
        r = brentq(g_of, min(rlo, rho0), max(rlo, rho0), xtol=1e-15*rho0, rtol=1e-15)
        p = K*r**g
        return r, p, rarefaction(st, p, g, sg)[1]

    def sample(xi):
        if xi < vs:
            w = waves['L']
            if w[0] == 'shock':
                return L if xi < w[2] else (w[1], ps, vs)
            if xi < w[2]:
                return L
            if xi > w[3]:
                return (w[1], ps, vs)
            return fan(L, -1, xi)
        w = waves['R']
        if w[0] == 'shock':
            return R if xi > w[2] else (w[1], ps, vs)
        if xi > w[2]:
            return R
        if xi < w[3]:
            return (w[1], ps, vs)
        return fan(R, +1, xi)
    return sample, dict(pstar=ps, vstar=vs, waves=waves)


def check_shock(st, ps, g, sg):
    """Rest-mass, momentum and energy fluxes across the shock in the lab frame:
    F - Vs U continuous."""
    rs, vs, Vs = shock(st, ps, g, sg)

    def UF(rho, p, v):
        W = 1/np.sqrt(1 - v*v)
        h = h_of(rho, p, g)
        D, S, E = rho*W, rho*h*W*W*v, rho*h*W*W - p
        return np.array([D, S, E]), np.array([D*v, S*v + p, S])
    U0, F0 = UF(*st)
    U1, F1 = UF(rs, ps, vs)
    jump = (F1 - Vs*U1) - (F0 - Vs*U0)
    return np.max(np.abs(jump)/np.abs(F0 - Vs*U0).max())


if __name__ == '__main__':
    g = 5.0/3.0
    for v in (0.0, 0.4):
        L, R = (10.0, 13.33, v), (1.0, 0.1, v)
        s, info = solve(L, R, g)
        print(f'v={v}: p*={info["pstar"]:.10f} v*={info["vstar"]:.10f}', info['waves'])
        for side, st, sg in (('L', L, -1), ('R', R, +1)):
            if info['waves'][side][0] == 'shock':
                print('  RH residual', side, check_shock(st, info['pstar'], g, sg))
    # boost covariance: v* and wave speeds of the v = 0.4 tube = relativistic sums
    s0, i0 = solve((10.0, 13.33, 0.0), (1.0, 0.1, 0.0), g)
    s1, i1 = solve((10.0, 13.33, 0.4), (1.0, 0.1, 0.4), g)
    add = lambda a, b: (a + b)/(1 + a*b)
    print('boost check v*:', add(i0['vstar'], 0.4) - i1['vstar'], ' p*:', i0['pstar'] - i1['pstar'])
    print('boost check R shock:', add(i0['waves']['R'][2], 0.4) - i1['waves']['R'][2])
