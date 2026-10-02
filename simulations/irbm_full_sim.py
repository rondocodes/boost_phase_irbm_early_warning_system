import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# IRBM-FOCUSED STAGE 3: FULL SIMULATION + PLOTS + TABLES
# ============================================================

OUT_DIR = Path("..")
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# CONSTANTS
# -----------------------------
MU = 398600.4418e9
R_E = 6378137.0
J2 = 1.08262668e-3
OMEGA_E = 7.2921159e-5

# -----------------------------
# CONSTELLATION CONFIG
# -----------------------------
ALT_KM = 1000.0
ALT = ALT_KM * 1000.0
PLANE_INCS_DEG = [30.0, 45.0, 70.0]
SATS_PER_PLANE = 4

# <<< REPLACE THESE WITH BEST VALUES FROM RAAN SEARCH >>>
D12_DEG = 120.0
D13_DEG = 60.0

# -----------------------------
# IRBM BOOST MODEL
# -----------------------------
BOOST_T = 210.0
BOOST_TARGET_ALT_M = 300e3
BOOST_V_TARGET = 4500.0
BOOST_MIN_DETECT_ALT = 15e3
PITCH_START_S = 10.0
PITCH_END_S = 90.0
PITCH_FINAL_DEG = 45.0

# -----------------------------
# SENSOR / DETECTION
# -----------------------------
MAX_STEER_DEG = 60.0
SLEW_RATE_DEG_S = 30.0
DWELL_S = 0.5
MAX_TRACKS_PER_SAT = 1

K_PLUME = 2e18
B_BG = 5e4
READ_NOISE = 150.0
COS_EXP = 1.0
SNR_THRESH = 4.0
ATM_K = 0.08

# -----------------------------
# SIMULATION
# -----------------------------
SIM_DURATION = 700.0
DT = 1.0
N_MC = 1000
RNG_SEED = 7

# -----------------------------
# SEA REGION MASK
# -----------------------------
SEA_CENTER_LAT = 8.75
SEA_CENTER_LON = 116.5
SEA_RADIUS_KM = 5500.0

LAT_MIN, LAT_MAX = -20, 50
LON_MIN, LON_MAX = 85, 175
DLAT, DLON = 5, 5

# -----------------------------
# PLOT SETTINGS
# -----------------------------
CMAP_P = "magma"
CMAP_T = "viridis"

# ============================================================
# HELPERS
# ============================================================
def rot_z(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c,  s, 0],
                     [-s,  c, 0],
                     [ 0,  0, 1]], dtype=float)

def rot_x(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0,  0],
                     [0, c,  s],
                     [0,-s,  c]], dtype=float)

def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n

def eci_to_ecef(r_eci, t):
    return rot_z(OMEGA_E * t) @ r_eci

def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = R_E + alt_m
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat)
    ], dtype=float)

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return (R_E / 1000.0) * c

def in_sea_mask(lat, lon):
    return haversine_km(lat, lon, SEA_CENTER_LAT, SEA_CENTER_LON) <= SEA_RADIUS_KM

# ============================================================
# ORBIT PROPAGATION WITH J2
# ============================================================
def accel_j2(r):
    x, y, z = r
    r2 = x*x + y*y + z*z
    rr = np.sqrt(r2)
    zz = z / rr
    fac = 1.5 * J2 * MU * (R_E**2) / (rr**5)
    return np.array([
        fac * x * (5*zz*zz - 1),
        fac * y * (5*zz*zz - 1),
        fac * z * (5*zz*zz - 3)
    ], dtype=float)

def ode_cowell_j2(_t, state):
    r = state[:3]
    v = state[3:]
    rr = np.linalg.norm(r)
    a2 = -MU * r / (rr**3)
    return np.concatenate([v, a2 + accel_j2(r)])

def orbital_period(alt_m):
    a = R_E + alt_m
    return 2*np.pi*np.sqrt(a**3 / MU)

def circular_orbit_state_eci(alt_m, inc_deg, raan_deg, phase_deg):
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    phase = np.radians(phase_deg)

    rmag = R_E + alt_m
    vmag = np.sqrt(MU / rmag)

    r0 = np.array([rmag, 0.0, 0.0])
    v0 = np.array([0.0, vmag, 0.0])

    r1 = rot_z(phase) @ r0
    v1 = rot_z(phase) @ v0

    r2 = rot_x(inc) @ r1
    v2 = rot_x(inc) @ v1

    r3 = rot_z(raan) @ r2
    v3 = rot_z(raan) @ v2
    return np.concatenate([r3, v3])

def propagate_satellite(state0, t_end, dt):
    n = int(np.floor(t_end/dt)) + 1
    t_eval = np.linspace(0.0, n*dt, n)
    t_eval = t_eval[t_eval <= t_end + 1e-9]
    t_eval[-1] = min(t_eval[-1], t_end)
    sol = solve_ivp(ode_cowell_j2, (0.0, t_end), state0, t_eval=t_eval, rtol=1e-8, atol=1e-8)
    if not sol.success:
        raise RuntimeError(sol.message)
    y = sol.y.T
    return sol.t, y[:, :3]

def interp_vec(t_grid, vec_grid, t):
    return np.array([
        np.interp(t, t_grid, vec_grid[:, 0]),
        np.interp(t, t_grid, vec_grid[:, 1]),
        np.interp(t, t_grid, vec_grid[:, 2]),
    ], dtype=float)

@dataclass
class SatEphem:
    name: str
    inc_deg: float
    raan_deg: float
    phase_deg: float
    t: np.ndarray
    r_eci: np.ndarray

def build_constellation():
    phases = [0.0, 90.0, 180.0, 270.0]
    raans = [0.0, D12_DEG, D13_DEG]
    sats = []
    idx = 0
    for p, (inc, raan) in enumerate(zip(PLANE_INCS_DEG, raans), start=1):
        for ph in phases:
            idx += 1
            sats.append((f"SAT{idx:02d}_P{p}", inc, raan, ph))
    return sats

def prepropagate_constellation():
    sats = build_constellation()
    T = orbital_period(ALT)
    t_end = T + SIM_DURATION
    ephems = []
    for name, inc, raan, ph in sats:
        st0 = circular_orbit_state_eci(ALT, inc, raan, ph)
        tt, r_eci = propagate_satellite(st0, t_end, DT)
        ephems.append(SatEphem(name, inc, raan, ph, tt, r_eci))
    return ephems, T

# ============================================================
# BOOSTER MODEL
# ============================================================
def enu_basis(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e = np.array([-np.sin(lon), np.cos(lon), 0.0])
    n = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    u = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    return e, n, u

def pitch_program(t):
    if t <= PITCH_START_S:
        return 0.0
    if t >= PITCH_END_S:
        return PITCH_FINAL_DEG
    f = (t - PITCH_START_S) / (PITCH_END_S - PITCH_START_S)
    return PITCH_FINAL_DEG * f

def booster_state_ecef(t, lat_deg, lon_deg, heading_deg=90.0):
    if t < 0 or t > BOOST_T:
        return None, None
    e, n, u = enu_basis(lat_deg, lon_deg)
    hd = np.radians(heading_deg)
    down = np.cos(hd)*n + np.sin(hd)*e

    tilt = np.radians(pitch_program(t))
    v_dir = unit(np.cos(tilt)*u + np.sin(tilt)*down)

    vmag = BOOST_V_TARGET * (t / BOOST_T)
    disp = 0.5 * vmag * t

    r0 = geodetic_to_ecef(lat_deg, lon_deg, 0.0)
    r = r0 + disp * v_dir
    alt = np.linalg.norm(r) - R_E

    if t > 0:
        desired_alt = BOOST_TARGET_ALT_M * (t / BOOST_T)**1.4
        current_alt = max(alt, 1.0)
        scale = (R_E + desired_alt) / (R_E + current_alt)
        r = r * scale
        alt = np.linalg.norm(r) - R_E

    return r, alt

# ============================================================
# DETECTION MODEL
# ============================================================
def segment_intersects_earth(r_sat, r_tgt, radius=R_E):
    d = r_tgt - r_sat
    a = np.dot(d, d)
    b = 2.0 * np.dot(r_sat, d)
    c = np.dot(r_sat, r_sat) - radius**2
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    s1 = (-b - np.sqrt(disc)) / (2*a)
    s2 = (-b + np.sqrt(disc)) / (2*a)
    return (0.0 <= s1 <= 1.0) or (0.0 <= s2 <= 1.0)

def off_nadir_angle(r_sat, r_tgt):
    rho = unit(r_tgt - r_sat)
    nadir = unit(-r_sat)
    return float(np.arccos(np.clip(np.dot(rho, nadir), -1.0, 1.0)))

def elevation_angle(r_sat, r_tgt):
    u_t = unit(r_tgt)
    los = unit(r_sat - r_tgt)
    z = np.arccos(np.clip(np.dot(u_t, los), -1.0, 1.0))
    return float(np.pi/2 - z)

def atm_trans(el):
    s = max(np.sin(max(el, 1e-3)), 1e-3)
    return float(np.exp(-ATM_K / s))

def snr_model(r_sat, r_tgt, alpha):
    rng = np.linalg.norm(r_tgt - r_sat)
    tau = atm_trans(elevation_angle(r_sat, r_tgt))
    cos_term = max(np.cos(alpha), 0.0)**COS_EXP
    S = (K_PLUME * tau * cos_term) / (rng**2 + 1.0)
    N = np.sqrt(max(S + B_BG + READ_NOISE**2, 1e-12))
    return float(S / N)

@dataclass
class SatPointing:
    b_ecef: np.ndarray
    dwell_s: float
    track_id: int

def steer_toward(b_old, b_cmd, max_delta):
    b_old = unit(b_old)
    b_cmd = unit(b_cmd)
    ang = np.arccos(np.clip(np.dot(b_old, b_cmd), -1.0, 1.0))
    if ang <= max_delta:
        return b_cmd, True
    frac = max_delta / ang
    return unit((1-frac)*b_old + frac*b_cmd), False

def detect_step_sat(r_sat, sat_state, targets):
    max_steer = np.radians(MAX_STEER_DEG)
    max_delta = np.radians(SLEW_RATE_DEG_S) * DT

    cands = []
    for tid, r_tgt in enumerate(targets):
        if r_tgt is None:
            continue
        if segment_intersects_earth(r_sat, r_tgt):
            continue
        alpha = off_nadir_angle(r_sat, r_tgt)
        if alpha > max_steer:
            continue
        b_cmd = unit(r_tgt - r_sat)
        snr = snr_model(r_sat, r_tgt, alpha)
        if snr < SNR_THRESH:
            continue
        cands.append((snr, tid, b_cmd))

    if not cands:
        sat_state.dwell_s = 0.0
        sat_state.track_id = -1
        return False, sat_state

    cands.sort(key=lambda x: x[0], reverse=True)
    _, tid, b_cmd = cands[0]

    b_new, aligned = steer_toward(sat_state.b_ecef, b_cmd, max_delta)
    sat_state.b_ecef = b_new

    if not aligned:
        sat_state.dwell_s = 0.0
        sat_state.track_id = -1
        return False, sat_state

    if sat_state.track_id == tid:
        sat_state.dwell_s += DT
    else:
        sat_state.track_id = tid
        sat_state.dwell_s = DT

    detected = sat_state.dwell_s >= DWELL_S
    return detected, sat_state

# ============================================================
# TRIAL SIMULATION
# ============================================================
def simulate_trial(ephems, T_orbit, lat, lon, n_targets, rng):
    t_offset = float(rng.uniform(0.0, T_orbit))
    sat_states = []

    for sat in ephems:
        r0_eci = interp_vec(sat.t, sat.r_eci, t_offset)
        r0_ecef = eci_to_ecef(r0_eci, t_offset)
        sat_states.append(SatPointing(unit(-r0_ecef), 0.0, -1))

    for t in np.arange(0.0, BOOST_T + 1e-9, DT):
        targets = []
        for k in range(n_targets):
            lon_k = lon + k * 15.0
            r_tgt, alt = booster_state_ecef(t, lat, lon_k)
            if r_tgt is None or alt < BOOST_MIN_DETECT_ALT:
                targets.append(None)
            else:
                targets.append(r_tgt)

        if all(x is None for x in targets):
            continue

        t_abs = t_offset + t
        for i, sat in enumerate(ephems):
            r_sat_eci = interp_vec(sat.t, sat.r_eci, t_abs)
            r_sat_ecef = eci_to_ecef(r_sat_eci, t_abs)
            detected, sat_states[i] = detect_step_sat(r_sat_ecef, sat_states[i], targets)
            if detected:
                return float(t)
    return None

def eval_point(ephems, T_orbit, lat, lon, n_targets, rng):
    det_times = []
    for _ in range(N_MC):
        dt_det = simulate_trial(ephems, T_orbit, lat, lon, n_targets, rng)
        if dt_det is not None:
            det_times.append(dt_det)

    p = len(det_times) / N_MC
    t_med = float(np.median(det_times)) if det_times else np.nan
    return p, t_med

# ============================================================
# TABLE EXPORTS
# ============================================================
def export_parameter_tables():
    arch = pd.DataFrame([
        {"Parameter": "Orbit type", "Value": "Circular LEO"},
        {"Parameter": "Altitude (km)", "Value": ALT_KM},
        {"Parameter": "Total satellites", "Value": 12},
        {"Parameter": "Orbital planes", "Value": 3},
        {"Parameter": "Satellites per plane", "Value": SATS_PER_PLANE},
        {"Parameter": "Inclinations (deg)", "Value": str(PLANE_INCS_DEG)},
        {"Parameter": "ΔRAAN12 (deg)", "Value": D12_DEG},
        {"Parameter": "ΔRAAN13 (deg)", "Value": D13_DEG},
        {"Parameter": "Max steering (deg)", "Value": MAX_STEER_DEG},
        {"Parameter": "Slew rate (deg/s)", "Value": SLEW_RATE_DEG_S},
        {"Parameter": "Monte Carlo trials per point", "Value": N_MC},
    ])
    arch.to_csv(OUT_DIR / "table_architecture_parameters.csv", index=False)

    sensor = pd.DataFrame([
        {"Parameter": "Plume radiance scaling K_PLUME", "Value": K_PLUME},
        {"Parameter": "Background radiance B_BG", "Value": B_BG},
        {"Parameter": "Read noise", "Value": READ_NOISE},
        {"Parameter": "Cosine angular exponent", "Value": COS_EXP},
        {"Parameter": "SNR threshold", "Value": SNR_THRESH},
        {"Parameter": "Atmospheric attenuation coefficient", "Value": ATM_K},
        {"Parameter": "Dwell time (s)", "Value": DWELL_S},
        {"Parameter": "Max concurrent tracks per sat", "Value": MAX_TRACKS_PER_SAT},
    ])
    sensor.to_csv(OUT_DIR / "table_sensor_parameters.csv", index=False)

    booster = pd.DataFrame([
        {"Parameter": "Threat class proxy", "Value": "IRBM-like"},
        {"Parameter": "Boost duration (s)", "Value": BOOST_T},
        {"Parameter": "Target burnout altitude (km)", "Value": BOOST_TARGET_ALT_M / 1000},
        {"Parameter": "Target burnout velocity (m/s)", "Value": BOOST_V_TARGET},
        {"Parameter": "Pitch start (s)", "Value": PITCH_START_S},
        {"Parameter": "Pitch end (s)", "Value": PITCH_END_S},
        {"Parameter": "Final pitch angle (deg)", "Value": PITCH_FINAL_DEG},
        {"Parameter": "Minimum detectable altitude (km)", "Value": BOOST_MIN_DETECT_ALT / 1000},
    ])
    booster.to_csv(OUT_DIR / "table_booster_parameters.csv", index=False)

# ============================================================
# PLOTTER
# ============================================================
def plot_heatmap(Z, lats, lons, title, cmap, vmin, vmax, fname):
    plt.figure(figsize=(12, 6))
    extent = [lons[0], lons[-1], lats[0], lats[-1]]
    im = plt.imshow(Z, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Value")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=220)
    plt.show()

# ============================================================
# MAIN
# ============================================================
def main():
    export_parameter_tables()

    rng = np.random.default_rng(RNG_SEED)
    print("Prepropagating constellation ...")
    ephems, T_orbit = prepropagate_constellation()
    print(f"Period ≈ {T_orbit:.1f} s")

    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, DLAT)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, DLON)

    P1 = np.full((len(lats), len(lons)), np.nan)
    T1 = np.full((len(lats), len(lons)), np.nan)
    P2 = np.full((len(lats), len(lons)), np.nan)
    T2 = np.full((len(lats), len(lons)), np.nan)

    rows = []

    valid_points = 0
    for i, lat in enumerate(lats):
        print(f"[LAT {i+1}/{len(lats)}] lat={lat:.1f}° | MC per point={N_MC}")
        for j, lon in enumerate(lons):
            if not in_sea_mask(lat, lon):
                continue

            valid_points += 1
            p1, tm1 = eval_point(ephems, T_orbit, lat, lon, n_targets=1, rng=rng)
            p2, tm2 = eval_point(ephems, T_orbit, lat, lon, n_targets=2, rng=rng)

            P1[i, j] = p1
            T1[i, j] = tm1
            P2[i, j] = p2
            T2[i, j] = tm2

            rows.append({
                "lat_deg": float(lat),
                "lon_deg": float(lon),
                "P_detect_single": p1,
                "T_median_single_s": tm1,
                "P_detect_multi": p2,
                "T_median_multi_s": tm2,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stage3_irbm_heatmap_data.csv", index=False)

    # ------------------------------------------------
    # HARDEST LAUNCH POINTS
    # ------------------------------------------------
    hardest = df.sort_values("P_detect_single").head(10)

    hardest_table = hardest[[
        "lat_deg",
        "lon_deg",
        "P_detect_single",
        "P_detect_multi",
        "T_median_single_s",
        "T_median_multi_s"
    ]]

    hardest_table.to_csv(OUT_DIR / "table_hardest_launch_points.csv", index=False)

    # Summary tables
    summary = pd.DataFrame([
        {
            "Scenario": "Single launch",
            "Mean P(detect)": float(np.nanmean(P1)),
            "Min P(detect)": float(np.nanmin(P1)),
            "Median detection time (s)": float(np.nanmedian(T1[np.isfinite(T1)])) if np.any(np.isfinite(T1)) else np.nan,
        },
        {
            "Scenario": "Two simultaneous launches",
            "Mean P(detect)": float(np.nanmean(P2)),
            "Min P(detect)": float(np.nanmin(P2)),
            "Median detection time (s)": float(np.nanmedian(T2[np.isfinite(T2)])) if np.any(np.isfinite(T2)) else np.nan,
        }
    ])
    summary.to_csv(OUT_DIR / "table_results_summary.csv", index=False)

    # ------------------------------------------------
    # WORST CASE SUMMARY TABLE
    # ------------------------------------------------
    worst_single_idx = np.nanargmin(P1)
    worst_multi_idx = np.nanargmin(P2)

    wi, wj = np.unravel_index(worst_single_idx, P1.shape)
    mi, mj = np.unravel_index(worst_multi_idx, P2.shape)

    worst_table = pd.DataFrame([
        {
            "Scenario": "Single launch",
            "Worst P(detect)": float(np.nanmin(P1)),
            "Worst latitude": float(lats[wi]),
            "Worst longitude": float(lons[wj])
        },
        {
            "Scenario": "Two launches",
            "Worst P(detect)": float(np.nanmin(P2)),
            "Worst latitude": float(lats[mi]),
            "Worst longitude": float(lons[mj])
        }
    ])

    worst_table.to_csv(OUT_DIR / "table_worst_case_summary.csv", index=False)

    # Regional summary subtable
    reg = pd.DataFrame([
        {
            "Region": "SEA 5500-km mask",
            "Valid grid points": valid_points,
            "Mean P_single": float(np.nanmean(P1)),
            "Mean P_multi": float(np.nanmean(P2)),
            "Min P_single": float(np.nanmin(P1)),
            "Min P_multi": float(np.nanmin(P2)),
        }
    ])
    reg.to_csv(OUT_DIR / "table_regional_summary.csv", index=False)

    print("\nResults summary:")
    print(summary.to_string(index=False))
    print("\nRegional summary:")
    print(reg.to_string(index=False))

    # Heatmaps
    plot_heatmap(
        P1, lats, lons,
        title=f"Single-launch P(detect)\n12 sats, inclinations {PLANE_INCS_DEG}, ΔΩ12={D12_DEG}°, ΔΩ13={D13_DEG}°",
        cmap=CMAP_P, vmin=0.0, vmax=1.0, fname="heatmap_P_single.png"
    )

    plot_heatmap(
        P2, lats, lons,
        title=f"Two-launch P(detect)\n12 sats, steer ±{MAX_STEER_DEG}°, slew {SLEW_RATE_DEG_S}°/s",
        cmap=CMAP_P, vmin=0.0, vmax=1.0, fname="heatmap_P_multi.png"
    )

    plot_heatmap(
        T1, lats, lons,
        title=f"Median detection time (single launch, conditional)",
        cmap=CMAP_T, vmin=0.0, vmax=BOOST_T, fname="heatmap_T_single.png"
    )

    plot_heatmap(
        T2, lats, lons,
        title=f"Median detection time (two launches, conditional)",
        cmap=CMAP_T, vmin=0.0, vmax=BOOST_T, fname="heatmap_T_multi.png"
    )

    # ------------------------------------------------
    # VULNERABILITY MAP
    # ------------------------------------------------
    vuln_map = np.zeros_like(P1)

    threshold = 0.5

    for i in range(P1.shape[0]):
        for j in range(P1.shape[1]):
            if np.isnan(P1[i, j]):
                vuln_map[i, j] = np.nan
            elif P1[i, j] < threshold:
                vuln_map[i, j] = 1
            else:
                vuln_map[i, j] = 0

    plot_heatmap(
        vuln_map,
        lats,
        lons,
        title=f"Vulnerability Map (P_detect < {threshold})",
        cmap="Reds",
        vmin=0,
        vmax=1,
        fname="heatmap_vulnerability.png"
    )

if __name__ == "__main__":
    main()