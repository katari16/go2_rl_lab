# Deploy-time command shaping — align-then-follow

Idea: instead of the current linear body-frame mapping (`v* = v_cmd + k * F̂_xy`), introduce an "align first, then walk forward" mode on top of the deployed policy. Purely deploy-time — no retraining needed.

## Motivation

Current behavior: a sideways push triggers sideways strafing, because the policy was trained to respond to body-frame force in the corresponding body-frame velocity axis. That is how the estimator is trained, but it is not how a human would follow a push. A more natural behavior: turn to face the push, then walk forward into it.

## Mode structure

Two modes, switched by the azimuth of the estimated planar force in the base frame:

```
φ = atan2(F̂_y, F̂_x)      # EMA-filtered force, not raw
```

- **Align mode** (`|φ| > φ_enter`): command `wz ∝ sign(φ)` (or `∝ φ` for smoothness); hold `vx = vy = 0`.
- **Follow mode** (`|φ| < φ_exit`): current compliance mapping active, `wz = 0`.

## Design notes

- **Hysteresis.** Don't use a single threshold. Example: `φ_enter = 30°`, `φ_exit = 20°`. Otherwise the mode flickers on the boundary as the estimate jitters.
- **Magnitude floor.** Only engage align when `|F̂_xy| > F_min` (~5–10 N). Below that, the azimuth is pure noise and the robot will spin on nothing.
- **Antipodal ambiguity.** A push from directly behind gives `φ ≈ ±π` and will flicker sign. Either bias toward the sign that agrees with the previous alignment decision, or clamp the yaw rate and let it settle to one side.
- **Yaw rate saturation.** Keep `wz_cmd` within the training distribution (±1 rad/s). A soft cap (e.g. `wz = clamp(k_yaw * φ, -0.7, 0.7)`) works fine.
- **Smooth the force, not the angle.** EMA on `F̂_x, F̂_y` and then compute φ. Avoids angle wrap-around.
- **What about `vx/vy` during align?** Zero them. The point of align is "don't walk yet, just turn." Otherwise you get a coupled turning + strafing transient that undoes the point of the mode.
- **Background check.** Mode decision runs every control step; the robot can re-enter align whenever the force direction drifts outside `φ_exit`.

## Open questions to resolve first

- **Estimator SNR on direction.** Does the force azimuth stay within ~±10–15° under a steady real-robot push? If it's more like ±45°, the scheme needs heavier smoothing and will feel sluggish. Check this with the existing real-robot recordings before implementing.
- **Yaw torque term.** When using the 6D estimator, should `τ̂_z` also feed into the yaw command in follow mode, or only the azimuth of the linear force drive heading? Probably the latter during align, but `τ̂_z`-based compliance during follow is still useful for pushes that rotate the robot without translating it.

## Parameter defaults (starting point)

```
phi_enter        = 30°  (0.52 rad)
phi_exit         = 20°  (0.35 rad)
F_min            = 5.0  N
k_yaw_align      = 1.2  (wz ≈ 1.2 * φ, clipped)
wz_clip          = 0.7  rad/s
ema_alpha_force  = 0.1  (existing)
```

## Implementation sketch

State machine lives in the deploy script, wrapping the existing compliance-mapping block. Roughly:

```python
state = "follow"
def update(force_ema, velocity_cmd):
    Fxy = force_ema[:2]
    F_mag = np.linalg.norm(Fxy)
    phi = np.arctan2(Fxy[1], Fxy[0])

    if F_mag < F_MIN:
        state = "follow"   # ignore align
        return velocity_cmd  # no compliance injection either

    if state == "follow" and abs(phi) > PHI_ENTER:
        state = "align"
    elif state == "align" and abs(phi) < PHI_EXIT:
        state = "follow"

    if state == "align":
        wz = np.clip(K_YAW_ALIGN * phi, -WZ_CLIP, WZ_CLIP)
        return np.array([0.0, 0.0, wz])
    else:
        # existing compliance mapping
        return velocity_cmd + k * np.array([Fxy[0], Fxy[1], 0.0])
```

Toggle via a joystick button (suggest L1) so we can A/B the two modes on the same run.
