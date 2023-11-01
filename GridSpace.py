
class GridSpace:
    def __init__(self, n_rows=4, n_cols=4, inaccessible=set([(1,1)]), success=set([(0,3)]), fail=set([(1,3)])):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.inaccessible = inaccessible
        self.success = success
        self.fail = fail
    
    def map_observation_to_repr(self, observation):
        return observation