class EstimatorBase:
	def __init__(self):
		self._est_state = None
	
	def solve(self):
		"""
		compute the unknown states for this time-step
		"""
		raise NotImplementedError
	
class WindEstimator(EstimatorBase):
	def __init__(self):
		super().__init__()
		