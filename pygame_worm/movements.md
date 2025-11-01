Physics-Based Kinematics:The worm's movement is driven by a "follow-the-leader" kinematic model in the Worm.update method. The head segment moves forward based on a speed derived from a sinusoidal undulation pattern (head_speed = params.forward_gain * (0.35 + 0.65 * abs(math.sin(omega * t_sec)))), and each subsequent segment follows the previous one, maintaining a fixed spacing (SEG_SPACING). This creates a smooth, wave-like motion typical of worm-like locomotion.
The direction of each segment is influenced by muscle activations, which adjust the segment's heading (seg.dir += turn * dt), simulating a curvature-based steering mechanism.

Muscle and Neuron Simulation:The MuscleFiber class models muscle contractions with a baseline activation, gain, and gamma parameters, which are modulated by neural "twitches" from the Neuron class. These twitches are stochastic, generated using a sigmoid function applied to a random Gaussian variable (sigmoid(self.k_gain * randn())), introducing variability in muscle contractions.
The Worm class uses left and right muscle activations to create differential turning. For example, a stronger left muscle activation causes the worm to turn left by increasing the curvature (turn = (left_a - right_a) * 0.25).
The central pattern generator (CPG) for locomotion is modeled by a sinusoidal wave (base = math.sin(phase) * params.wave_amp), which mimics the rhythmic undulation seen in real worms.

Sensor-Driven Steering:The worm uses simulated sensors to detect obstacles via raycasting (maze.raycast). The difference in distances detected by left and right sensors (dL and dR) creates a steering bias (bias = params.steer_gain * (dR - dL) / max(1.0, params.sensor_range)), allowing the worm to navigate around obstacles in the maze.
This sensor feedback dynamically adjusts the worm's path, making its movement responsive to the environment rather than following a fixed trajectory.

Stochastic Neural Activity:The Neuron class introduces randomness in muscle activations through periodic twitches, which are triggered at intervals (interval_ms) and vary in amplitude and timing. This randomness adds lifelike variability to the worm's movements, preventing it from being entirely deterministic.
Background neurons (bg_neuron) occasionally trigger random mid-body segments, adding spontaneous twitches that mimic natural irregularities in worm movement.

Collision Handling:The worm interacts with the maze's boundaries and obstacles through collision detection (maze.collide_circle). When the head collides, it is pushed back slightly and nudged to rotate away, ensuring realistic interaction with the environment.

Food Attraction: The worm's behavior is influenced by food attraction through a sensor system in the Worm class. The worm uses simulated sensors to detect food items in the maze via raycasting (maze.raycast). The difference in distances detected by left and right sensors (dL and dR) creates a steering bias (bias = params.steer_gain * (dR - dL) / max(1.0, params.sensor_range)), allowing the worm to navigate toward food sources.
This sensor feedback dynamically adjusts the worm's path, making its movement responsive to the environment rather than following a fixed trajectory.
