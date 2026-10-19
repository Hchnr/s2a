"""On-demand rendering with the same RGB convention as robosuite observables."""


def render_current_cameras(env, resolution):
    import robosuite.macros as macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    return {
        f"{name}_image": env.sim.render(camera_name=name, width=resolution, height=resolution, depth=False)[
            ::convention
        ].copy()
        for name in ("agentview", "robot0_eye_in_hand")
    }
