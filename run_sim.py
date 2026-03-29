# Main 6DOF simulation script.

import torch

from missilerl.sim6dof import Missile6DOF, MissileParams, SimulationWorld, Target


def weaving_trajectory(time_s: float, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vx = -70.0
    vy = 35.0 * torch.sin(torch.tensor(0.22 * time_s))
    vz = 12.0 * torch.cos(torch.tensor(0.11 * time_s))
    vel = torch.tensor([vx, vy.item(), vz.item()], dtype=torch.float32)
    return pos + vel * 0.02, vel


def main() -> None:
    params = MissileParams()

    missile = Missile6DOF(
        params=params,
        position_world=torch.tensor([0.0, 0.0, 1200.0]),
        velocity_world=torch.tensor([280.0, 0.0, 0.0]),
        quat_body_to_world=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        omega_body=torch.zeros(3),
    )

    # Waypoint target mode.
    target = Target(
        position_world=torch.tensor([4200.0, 450.0, 1050.0]),
        waypoints=torch.tensor(
            [
                [4200.0, 450.0, 1050.0],
                [4700.0, 150.0, 1100.0],
                [4500.0, -400.0, 980.0],
            ]
        ),
        waypoint_speed=130.0,
        loop_waypoints=True,
    )

    # Function target mode.
    # target = Target(
    #     position_world=torch.tensor([4200.0, 450.0, 1050.0]),
    #     trajectory_fn=weaving_trajectory,
    # )

    world = SimulationWorld(missile=missile, target=target, dt=0.02)

    for k in range(1000):
        rel = target.pos - missile.pos

        # Simple demo guidance.
        fin_top = torch.clamp(-0.0002 * rel[2], -0.15, 0.15)
        fin_bottom = -fin_top
        fin_right = torch.clamp(0.0002 * rel[1], -0.15, 0.15)
        fin_left = -fin_right
        throttle = torch.tensor(1.0 if k < 120 else (0.7 if k < 400 else 0.4))

        control = torch.stack([fin_top, fin_bottom, fin_left, fin_right, throttle])
        out = world.step(control)

        if k % 50 == 0 or out["done"]:
            miss = torch.linalg.norm(target.pos - missile.pos).item()
            print(f"step={k:04d} t={out['time']:.2f}s miss={miss:.1f}m hit={out['hit']} oob={out['oob']}")

        if out["done"]:
            break


if __name__ == "__main__":
    main()
