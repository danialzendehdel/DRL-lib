


class Journal:
    def __init__(self, action_bounds):
        self.entries = []

        self.action_bound_lower = action_bounds[0]
        self.action_bound_upper = action_bounds[1]


    
    def _actor_debug(self, sample_action):

        bound_check = (sample_action > self.action_bound_lower) or (sample_action < self.action_bound_upper) 

        if bound_check:
            self.entries.append(f"Action out of bounds: {sample_action}")




