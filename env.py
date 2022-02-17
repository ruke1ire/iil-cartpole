from continuous_cartpole import *

class MaxStepContinuousCartPoleEnv(ContinuousCartPoleEnv):
    def __init__(self, max_steps = 3000, gamma = 0.95):
        super().__init__()
        self.step_no = 0
        self.max_steps = max_steps
        self.gamma = gamma

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        self.step_no += 1
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        max_steps_reached = self.step_no >= self.max_steps
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians \
            or max_steps_reached
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            if(max_steps_reached):
                reward = 1*(1-self.gamma)
            else:
                reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.step_no = 0
        return np.array(self.state)