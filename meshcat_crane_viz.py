import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time

def visualize_crane_meshcat(sol_closed, params, t_eval=None, playback_speed=1.0, max_fps=30.0):
    """
    Visualize the crane simulation using Meshcat.

    Parameters:
    - sol_closed: Solution object from solve_ivp for closed-loop simulation
    - params: Dictionary with system parameters
    - t_eval: Time evaluation points. If None, sol_closed.t is used.
    - playback_speed: 1.0 means real time, 2.0 means twice as fast.
    - max_fps: Maximum visualization update rate.
    """
    if playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive")

    t_grid = np.asarray(t_eval if t_eval is not None else sol_closed.t, dtype=float)
    n_states = sol_closed.y.shape[1]
    if len(t_grid) != n_states:
        t_grid = np.asarray(sol_closed.t, dtype=float)
    if len(t_grid) != n_states:
        raise ValueError("t_eval must have the same number of samples as sol_closed.y")

    vis = meshcat.Visualizer()
    vis.open()

    def camera_pose(eye, target, up=np.array([0.0, 0.0, 1.0])):
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / np.linalg.norm(right)

        true_up = np.cross(right, forward)

        T = np.eye(4)
        T[:3, 0] = right
        T[:3, 1] = true_up
        T[:3, 2] = -forward   # důležité: kamera kouká po -Z
        T[:3, 3] = eye
        return T

    # Initial camera pose hard-set to a wide top-front view of the crane
    eye = np.array([10.0, -10.0, 12.0])
    target = np.array([0.8, 0.0, 1.5])
    cam_pose = camera_pose(eye, target)
    vis["Cameras/default/pose"].set_transform(cam_pose)

    # Optionally lock view by resetting camera controls
    vis["Cameras/default/far"].set_property("value", 1000.0)

    # Global scale so the whole visualization is 5 times smaller
    scale = 0.2

    # Set ground plane
    ground_geom = g.Box([20.0 * scale, 20.0 * scale, 0.1 * scale])
    vis["ground"].set_object(ground_geom, g.MeshLambertMaterial(color=0xcccccc))
    vis["ground"].set_transform(tf.translation_matrix([0, 0, -0.1 * scale]))

    # Vertical boom (boom1): fixed, from ground (z=0) to z=8
    boom1_geom = g.Box([0.5 * scale, 0.5 * scale, 8.0 * scale])
    vis["boom1"].set_object(boom1_geom, g.MeshLambertMaterial(color=0x8b4513))
    vis["boom1"].set_transform(tf.translation_matrix([0, 0, 4.0 * scale]))  # centered at height 4.0 (from 0 to 8)

    # Horizontal boom (boom2): length 5m, starts at top of boom1 (z=8)
    boom2_length = 5.0 * scale
    boom2_geom = g.Box([boom2_length, 0.2 * scale, 0.2 * scale])
    vis["boom2"].set_object(boom2_geom, g.MeshLambertMaterial(color=0xffaa00))

    # Trolley (vozík): box that moves along horizontal boom
    trolley_geom = g.Box([0.3 * scale, 0.3 * scale, 0.2 * scale])
    vis["trolley"].set_object(trolley_geom, g.MeshLambertMaterial(color=0x0000ff))

    # Payload (břemeno): sphere
    payload_geom = g.Sphere(0.4 * scale)
    vis["payload"].set_object(payload_geom, g.MeshLambertMaterial(color=0xff0000))

    # Function to update positions
    def update_crane(state, params):
        r, phi, alpha, beta = state[:4]
        l = params["l"]

        # Boom1 (vertical): fixed at origin - no update needed
        # Boom2 (horizontal): rotates around the top of boom1 by phi
        boom2_transform = (
            tf.translation_matrix([0.0, 0.0, 8.0 * scale])
            @ tf.rotation_matrix(phi, [0.0, 0.0, 1.0])
            @ tf.translation_matrix([boom2_length / 2.0, 0.0, 0.0])
        )
        vis["boom2"].set_transform(boom2_transform)

        # Trolley position: moves along the rotated boom2 in the local x-direction
        p_v = np.array([r * np.cos(phi), r * np.sin(phi), 8.0]) * scale
        vis["trolley"].set_transform(tf.translation_matrix(p_v))

        # Payload position: hanging on cable from trolley (tilting in x and y directions)
        # alpha: tilt in x-direction, beta: tilt in y-direction
        p_payload = p_v + l * scale * np.array([
            np.sin(alpha) * np.cos(beta),
            np.sin(beta),
            -np.cos(alpha) * np.cos(beta)
        ])
        vis["payload"].set_transform(tf.translation_matrix(p_payload))

        # Cable: update position and orientation to connect trolley to payload
        cable_vector = p_payload - p_v
        cable_length = np.linalg.norm(cable_vector)
        cable_midpoint = (p_v + p_payload) / 2.0

        # Update cable geometry with correct length
        if cable_length > 1e-6:
            cable_geom = g.Cylinder(radius=0.02, height=cable_length)
            vis["cable"].set_object(cable_geom, g.MeshLambertMaterial(color=0x000000))
            
            cable_direction = cable_vector / cable_length
            # Y-axis of cylinder should point along cable direction
            y_axis = np.array([0.0, 1.0, 0.0])
            
            # Compute rotation axis and angle
            if np.abs(np.dot(y_axis, cable_direction) - 1.0) < 1e-6:
                # Already aligned
                rot_matrix = np.eye(3)
            elif np.abs(np.dot(y_axis, cable_direction) + 1.0) < 1e-6:
                # Opposite direction
                rot_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            else:
                rotation_axis = np.cross(y_axis, cable_direction)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.clip(np.dot(y_axis, cable_direction), -1, 1))
                
                # Rodrigues rotation formula
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                rot_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
            
            # Create full transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = cable_midpoint
            vis["cable"].set_transform(T)

    frame_step = 1
    if max_fps is not None and max_fps > 0.0 and len(t_grid) > 1:
        positive_dt = np.diff(t_grid)
        positive_dt = positive_dt[positive_dt > 0.0]
        if len(positive_dt) > 0:
            sample_dt = np.median(positive_dt)
            min_sim_frame_dt = playback_speed / max_fps
            frame_step = max(1, int(np.ceil(min_sim_frame_dt / sample_dt)))

    start_wall_time = time.perf_counter()
    start_sim_time = t_grid[0]

    # Animate at the same pace as simulation time by default.
    for i in range(0, len(t_grid), frame_step):
        target_wall_time = (t_grid[i] - start_sim_time) / playback_speed
        sleep_time = target_wall_time - (time.perf_counter() - start_wall_time)
        if sleep_time > 0.0:
            time.sleep(sleep_time)

        state = sol_closed.y[:, i]
        update_crane(state, params)

    update_crane(sol_closed.y[:, -1], params)


