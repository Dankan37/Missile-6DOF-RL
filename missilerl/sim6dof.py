# 6DOF missile and world stepper.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


TargetTrajectoryFn = Callable[[float, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q), min=1e-9)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    # Quaternion to rotation matrix.
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return torch.tensor(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=q.dtype,
        device=q.device,
    )


def _quat_rate(q: torch.Tensor, omega_body: torch.Tensor) -> torch.Tensor:
    # Quaternion derivative from body rates.
    p, q_rate, r = omega_body
    omega_mat = torch.tensor(
        [
            [0.0, -p, -q_rate, -r],
            [p, 0.0, r, -q_rate],
            [q_rate, -r, 0.0, p],
            [r, q_rate, -p, 0.0],
        ],
        dtype=q.dtype,
        device=q.device,
    )
    return 0.5 * omega_mat @ q


@dataclass
class MissileParams:
    mass: float = 85.0
    length: float = 2.6
    diameter: float = 0.18

    # Atmosphere constants.
    rho: float = 1.225
    gravity: float = 9.81
    speed_of_sound: float = 340.0

    # Fin control properties.
    max_fin_deflection_rad: float = 0.35
    fin_lift_coeff: float = 300.0
    max_rate_rad_s: float = 12.0

    # Aerodynamic drag terms.
    area_ref: float = 0.025
    cd0: float = 0.06
    cd_base: float = 0.025
    cd_alpha_gain: float = 1.4
    induced_drag_k: float = 0.22
    wave_drag_gain: float = 0.18
    wave_drag_mach_start: float = 0.85
    wave_drag_mach_peak: float = 1.25

    # Motor booster and sustainer.
    booster_thrust: float = 15000.0
    sustainer_thrust: float = 5200.0
    booster_burn_time: float = 2.2
    sustainer_burn_time: float = 8.5
    throttle_time_constant: float = 0.15


class Missile6DOF:
    # 6DOF missile model.

    def __init__(
        self,
        params: MissileParams,
        position_world: torch.Tensor,
        velocity_world: torch.Tensor,
        quat_body_to_world: torch.Tensor,
        omega_body: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        self.params = params
        self.device = device or torch.device("cpu")
        self.dtype = torch.float32

        self.pos = position_world.to(self.device, self.dtype)
        self.vel = velocity_world.to(self.device, self.dtype)
        self.quat = _normalize_quat(quat_body_to_world.to(self.device, self.dtype))
        self.omega_body = omega_body.to(self.device, self.dtype)

        self.time_since_launch = 0.0
        self.throttle_state = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        radius = 0.5 * params.diameter
        ix = 0.5 * params.mass * radius * radius
        iy = (1.0 / 12.0) * params.mass * (3.0 * radius * radius + params.length * params.length)
        iz = iy

        self.inertia_body = torch.tensor([ix, iy, iz], device=self.device, dtype=self.dtype)

    def _thrust_available(self, t: float) -> float:
        if t < self.params.booster_burn_time:
            return self.params.booster_thrust
        if t < (self.params.booster_burn_time + self.params.sustainer_burn_time):
            return self.params.sustainer_thrust
        return 0.0

    def _wave_drag_coefficient(self, mach: torch.Tensor) -> torch.Tensor:
        m0 = self.params.wave_drag_mach_start
        m1 = self.params.wave_drag_mach_peak
        x = torch.clamp((mach - m0) / max(1e-6, m1 - m0), min=0.0, max=1.0)
        rise = x * x * (3.0 - 2.0 * x)  # smoothstep rise near transonic region
        supersonic_tail = torch.clamp(mach - m1, min=0.0)
        return self.params.wave_drag_gain * (rise + 0.15 * supersonic_tail)

    def _drag_coefficient(self, speed: torch.Tensor, vel_body: torch.Tensor) -> torch.Tensor:
        mach = speed / max(self.params.speed_of_sound, 1e-6)
        v_lat = torch.linalg.norm(vel_body[1:])
        alpha = torch.atan2(v_lat, torch.clamp(torch.abs(vel_body[0]), min=1e-6))

        cd_parasite = self.params.cd0 + self.params.cd_base
        cd_alpha = self.params.cd_alpha_gain * (alpha * alpha)
        cl_eff = self.params.fin_lift_coeff * alpha * 0.01
        cd_induced = self.params.induced_drag_k * cl_eff * cl_eff
        cd_wave = self._wave_drag_coefficient(mach)
        return cd_parasite + cd_alpha + cd_induced + cd_wave

    def _update_throttle_state(self, throttle_cmd: torch.Tensor, dt: float) -> torch.Tensor:
        tau = max(self.params.throttle_time_constant, 1e-4)
        self.throttle_state = self.throttle_state + (dt / tau) * (throttle_cmd - self.throttle_state)
        self.throttle_state = torch.clamp(self.throttle_state, min=0.0, max=1.0)
        return self.throttle_state

    def step(self, control: torch.Tensor, dt: float) -> None:
        # Advance one timestep.
        assert control.shape == (5,), "control must be shape (5,)"
        control = control.to(self.device, self.dtype)

        fin_cmd = torch.clamp(
            control[:4],
            min=-self.params.max_fin_deflection_rad,
            max=self.params.max_fin_deflection_rad,
        )
        throttle_cmd = torch.clamp(control[4], min=0.0, max=1.0)
        throttle = self._update_throttle_state(throttle_cmd, dt)

        rot_b2w = _quat_to_rotmat(self.quat)
        rot_w2b = rot_b2w.T

        vel_body = rot_w2b @ self.vel
        speed = torch.linalg.norm(self.vel)
        qbar = 0.5 * self.params.rho * speed * speed

        thrust_available = self._thrust_available(self.time_since_launch)
        thrust_body = torch.tensor(
            [thrust_available * throttle, 0.0, 0.0],
            device=self.device,
            dtype=self.dtype,
        )

        drag_body = torch.zeros(3, device=self.device, dtype=self.dtype)
        if speed > 1e-5:
            cd = self._drag_coefficient(speed, vel_body)
            drag_mag = qbar * cd * self.params.area_ref
            drag_body = -drag_mag * (vel_body / torch.clamp(torch.linalg.norm(vel_body), min=1e-6))

        # Differential fins generate lateral force.
        fz = self.params.fin_lift_coeff * qbar * (fin_cmd[0] - fin_cmd[1])
        fy = self.params.fin_lift_coeff * qbar * (fin_cmd[3] - fin_cmd[2])
        fin_force_body = torch.tensor([0.0, fy, fz], device=self.device, dtype=self.dtype)

        force_world = rot_b2w @ (thrust_body + drag_body + fin_force_body)
        gravity_world = torch.tensor(
            [0.0, 0.0, -self.params.mass * self.params.gravity],
            device=self.device,
            dtype=self.dtype,
        )
        total_force_world = force_world + gravity_world

        acc_world = total_force_world / self.params.mass
        self.vel = self.vel + acc_world * dt
        self.pos = self.pos + self.vel * dt

        # Tail lever arm moments.
        lever_arm_x = -0.45 * self.params.length
        moment_body = torch.tensor(
            [
                0.0,
                lever_arm_x * fz,
                -lever_arm_x * fy,
            ],
            device=self.device,
            dtype=self.dtype,
        )

        omega_cross_iw = torch.cross(self.omega_body, self.inertia_body * self.omega_body, dim=0)
        omega_dot = (moment_body - omega_cross_iw) / self.inertia_body
        self.omega_body = self.omega_body + omega_dot * dt
        self.omega_body = torch.clamp(
            self.omega_body,
            min=-self.params.max_rate_rad_s,
            max=self.params.max_rate_rad_s,
        )

        self.quat = _normalize_quat(self.quat + _quat_rate(self.quat, self.omega_body) * dt)
        self.time_since_launch += dt

    def state(self) -> Dict[str, torch.Tensor | float]:
        return {
            "pos": self.pos.clone(),
            "vel": self.vel.clone(),
            "quat": self.quat.clone(),
            "omega_body": self.omega_body.clone(),
            "time_since_launch": self.time_since_launch,
            "throttle_state": self.throttle_state.clone(),
        }


class Target:
    # Target point mass model.

    def __init__(
        self,
        position_world: torch.Tensor,
        velocity_world: Optional[torch.Tensor] = None,
        waypoints: Optional[torch.Tensor] = None,
        waypoint_speed: float = 120.0,
        loop_waypoints: bool = False,
        trajectory_fn: Optional[TargetTrajectoryFn] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.dtype = torch.float32
        self.pos = position_world.to(self.device, self.dtype)
        self.vel = (velocity_world if velocity_world is not None else torch.zeros(3)).to(self.device, self.dtype)

        self.time = 0.0
        self.trajectory_fn = trajectory_fn
        self.waypoints = waypoints.to(self.device, self.dtype) if waypoints is not None else None
        self.waypoint_speed = waypoint_speed
        self.loop_waypoints = loop_waypoints
        self._waypoint_index = 0

        if self.waypoints is not None and self.waypoints.ndim != 2:
            raise ValueError("waypoints must have shape [N, 3]")

    def _step_waypoints(self, dt: float) -> None:
        if self.waypoints is None or self.waypoints.shape[0] == 0:
            self.pos = self.pos + self.vel * dt
            return

        target_wp = self.waypoints[self._waypoint_index]
        delta = target_wp - self.pos
        dist = torch.linalg.norm(delta)

        if dist < 1e-3:
            if self._waypoint_index < (self.waypoints.shape[0] - 1):
                self._waypoint_index += 1
            elif self.loop_waypoints:
                self._waypoint_index = 0
            self.vel = torch.zeros(3, device=self.device, dtype=self.dtype)
            return

        direction = delta / torch.clamp(dist, min=1e-6)
        self.vel = direction * self.waypoint_speed

        step_dist = self.waypoint_speed * dt
        if step_dist >= dist:
            self.pos = target_wp
        else:
            self.pos = self.pos + self.vel * dt

    def step(self, dt: float) -> None:
        self.time += dt

        if self.trajectory_fn is not None:
            new_pos, new_vel = self.trajectory_fn(self.time, self.pos.clone())
            self.pos = new_pos.to(self.device, self.dtype)
            self.vel = new_vel.to(self.device, self.dtype)
            return

        if self.waypoints is not None:
            self._step_waypoints(dt)
            return

        self.pos = self.pos + self.vel * dt


class SimulationWorld:
    # Missile and target world stepper.

    def __init__(
        self,
        missile: Missile6DOF,
        target: Optional[Target] = None,
        dt: float = 0.01,
        world_bounds: torch.Tensor | None = None,
        hit_radius: float = 10.0,
    ):
        self.missile = missile
        self.target = target
        self.dt = dt
        self.hit_radius = hit_radius
        self.time = 0.0
        self.world_bounds = (
            world_bounds.to(dtype=torch.float32)
            if world_bounds is not None
            else torch.tensor([20000.0, 20000.0, 20000.0], dtype=torch.float32)
        )

    def is_oob(self) -> bool:
        p = torch.abs(self.missile.pos)
        return bool(torch.any(p > self.world_bounds)) or bool(self.missile.pos[2] < 0.0)

    def has_hit_target(self) -> bool:
        if self.target is None:
            return False
        dist = torch.linalg.norm(self.target.pos - self.missile.pos)
        return bool(dist <= self.hit_radius)

    def step(self, missile_control: torch.Tensor) -> Dict[str, object]:
        self.missile.step(missile_control, self.dt)
        if self.target is not None:
            self.target.step(self.dt)

        self.time += self.dt
        oob = self.is_oob()
        hit = self.has_hit_target()
        done = oob or hit

        out = {
            "time": self.time,
            "missile": self.missile.state(),
            "oob": oob,
            "hit": hit,
            "done": done,
        }
        if self.target is not None:
            out["target_pos"] = self.target.pos.clone()
            out["target_vel"] = self.target.vel.clone()
        return out


if __name__ == "__main__":
    params = MissileParams()
    missile = Missile6DOF(
        params=params,
        position_world=torch.tensor([0.0, 0.0, 1200.0]),
        velocity_world=torch.tensor([280.0, 0.0, 0.0]),
        quat_body_to_world=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        omega_body=torch.zeros(3),
    )

    target = Target(
        position_world=torch.tensor([4200.0, 450.0, 1050.0]),
        waypoints=torch.tensor(
            [
                [4200.0, 450.0, 1050.0],
                [4500.0, -200.0, 1020.0],
                [3900.0, -450.0, 1080.0],
            ]
        ),
        waypoint_speed=110.0,
        loop_waypoints=True,
    )

    world = SimulationWorld(missile=missile, target=target, dt=0.02)

    for k in range(500):
        rel = target.pos - missile.pos

        # Simple proportional demo control.
        fin_top = torch.clamp(-0.00018 * rel[2], -0.12, 0.12)
        fin_bottom = -fin_top
        fin_right = torch.clamp(0.00018 * rel[1], -0.12, 0.12)
        fin_left = -fin_right
        throttle = torch.tensor(1.0 if k < 120 else 0.7)

        control = torch.stack([fin_top, fin_bottom, fin_left, fin_right, throttle])
        out = world.step(control)

        if k % 30 == 0 or out["done"]:
            miss = torch.linalg.norm(target.pos - missile.pos).item()
            print(f"t={out['time']:.2f}s miss={miss:.1f}m hit={out['hit']} oob={out['oob']}")
        if out["done"]:
            break
