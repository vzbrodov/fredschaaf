"""Propagation of Gaia FPR heliocentric state vectors with ASSIST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.time import Time


FPR_REFERENCE_JD = 2_455_197.5
FPR_AU_METRES = 149_597_871_473.216
IAU_AU_METRES = 149_597_870_700.0
LB = 1.550519768e-8


def fpr_state_to_tdb(state_fpr: np.ndarray) -> np.ndarray:
    """Apply the published FPR scale correction and TCB-to-TDB scaling.

    Gaia Collaboration (2023, Eqs. 9--10) specifies that both position and
    velocity are first multiplied by ``au_FPR/au`` to obtain a TCB-compatible
    state. The TDB position then receives an additional ``1-LB`` factor while
    velocity does not.
    """
    state = np.asarray(state_fpr, dtype=float)
    if state.shape != (6,):
        raise ValueError("FPR state vector must contain six elements")
    rho = FPR_AU_METRES / IAU_AU_METRES
    result = state * rho
    result[:3] *= 1.0 - LB
    return result


def tcb_offset_to_tdb_jd(offset_days: np.ndarray) -> np.ndarray:
    """Convert Gaia JD(TCB)-J2010 offsets to absolute JD(TDB)."""
    return np.asarray(
        Time(FPR_REFERENCE_JD + np.asarray(offset_days, dtype=float), format="jd", scale="tcb").tdb.jd
    )


def propagate_fpr_state(
    state_fpr: np.ndarray,
    state_epoch_tcb_offset_days: float,
    observation_epoch_tcb_offset_days: np.ndarray,
    gaia_barycentric_fpr_au: np.ndarray,
    *,
    planets_path: str | Path,
    asteroids_path: str | Path,
    light_time_iterations: int = 3,
    observer_positions_are_tcb: bool = True,
    solar_light_deflection_mode: str = "absolute",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return light-time-corrected model RA/Dec and emission JD(TDB).

    The returned direction is geometric in ICRF from Gaia at reception to the
    asteroid at emission. Relativistic aberration is already removed from FPR
    coordinates. Solar-system light deflection is deliberately not added here
    and must be assessed separately at sub-mas precision.
    """
    import assist
    import rebound

    epochs_tdb = tcb_offset_to_tdb_jd(observation_epoch_tcb_offset_days)
    epoch0_tdb = float(tcb_offset_to_tdb_jd([state_epoch_tcb_offset_days])[0])
    ephem = assist.Ephem(str(planets_path), str(asteroids_path))
    state_tdb = fpr_state_to_tdb(state_fpr)
    t0 = epoch0_tdb - ephem.jd_ref
    sun = ephem.get_particle("Sun", t0)
    initial = np.array(
        [sun.x, sun.y, sun.z, sun.vx, sun.vy, sun.vz], dtype=float
    ) + state_tdb

    def positions_at(jd_tdb: np.ndarray) -> np.ndarray:
        targets = np.asarray(jd_tdb, dtype=float) - ephem.jd_ref
        result = np.empty((len(targets), 3), dtype=float)
        before = np.where(targets < t0)[0]
        after = np.where(targets >= t0)[0]
        for indices, reverse in ((before, True), (after, False)):
            if not len(indices):
                continue
            simulation = rebound.Simulation()
            simulation.t = t0
            simulation.add(
                x=initial[0], y=initial[1], z=initial[2],
                vx=initial[3], vy=initial[4], vz=initial[5],
            )
            extras = assist.Extras(simulation, ephem)
            order = indices[np.argsort(targets[indices])]
            if reverse:
                order = order[::-1]
            for index in order:
                extras.integrate_or_interpolate(float(targets[index]))
                result[index] = simulation.particles[0].xyz
        return result

    gaia = np.asarray(gaia_barycentric_fpr_au, dtype=float)
    if gaia.shape != (len(epochs_tdb), 3):
        raise ValueError("Gaia barycentric positions must have shape (N, 3)")
    # BCRS spatial coordinates receive the same TCB-to-TDB scale factor.
    gaia_tdb = gaia * (1.0 - LB) if observer_positions_are_tcb else gaia
    emission_tdb = epochs_tdb.copy()
    asteroid = positions_at(emission_tdb)
    for _ in range(light_time_iterations):
        distance = np.linalg.norm(asteroid - gaia_tdb, axis=1)
        emission_tdb = epochs_tdb - distance / ephem.c_AU_per_day
        asteroid = positions_at(emission_tdb)

    direction = asteroid - gaia_tdb
    distance = np.linalg.norm(direction, axis=1)
    unit = direction / distance[:, None]
    if solar_light_deflection_mode not in {"absolute", "differential", "none"}:
        raise ValueError("solar_light_deflection_mode must be absolute, differential, or none")
    if solar_light_deflection_mode != "none":
        import erfa

        sun_positions = np.empty_like(gaia_tdb)
        for index, jd_tdb in enumerate(epochs_tdb):
            sun = ephem.get_particle("Sun", float(jd_tdb - ephem.jd_ref))
            sun_positions[index] = sun.xyz
        sun_to_source = asteroid - sun_positions
        sun_to_source /= np.linalg.norm(sun_to_source, axis=1)[:, None]
        sun_to_observer = gaia_tdb - sun_positions
        observer_distance = np.linalg.norm(sun_to_observer, axis=1)
        sun_to_observer /= observer_distance[:, None]
        geometric_unit = unit.copy()
        finite_deflected = erfa.ld(
            1.0,
            geometric_unit,
            sun_to_source,
            sun_to_observer,
            observer_distance,
            1e-6,
        )
        if solar_light_deflection_mode == "absolute":
            unit = finite_deflected
        else:
            infinite_deflected = erfa.ld(
                1.0,
                geometric_unit,
                geometric_unit,
                sun_to_observer,
                observer_distance,
                1e-6,
            )
            unit = finite_deflected - (infinite_deflected - geometric_unit)
        unit /= np.linalg.norm(unit, axis=1)[:, None]
    ra = np.rad2deg(np.arctan2(unit[:, 1], unit[:, 0])) % 360.0
    dec = np.rad2deg(np.arcsin(unit[:, 2]))
    return ra, dec, emission_tdb
