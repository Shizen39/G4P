def get_action(observation, states):
	if observation[0] > states[0][2]:
		if observation[3] > states[3][0]:
			if observation[3] > states[3][0]:
				action = 0

			else:
				if observation[3] > states[3][0]:
					action = 1

	else:
		if observation[0] > states[0][2]:
			if observation[3] > states[3][0]:
				if observation[3] > states[3][0]:
					action = 0

				else:
					action = 1

		else:
			if observation[1] > states[1][0]:
				if observation[0] > states[0][2]:
					action = 1

				else:
					action = 1

			else:
				if observation[3] > states[3][0]:
					action = 0

				else:
					action = 1

	return action