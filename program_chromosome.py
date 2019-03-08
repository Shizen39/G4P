def get_action(observation, states):
	if observation[1] > states[1][2]:
		if observation[1] > states[1][0]:
			action = 0

		else:
			if observation[1] > states[1][2]:
				if observation[1] > states[1][0]:
					action = 0

				else:
					action = 1

	if observation[3] > states[3][1]:
		if observation[1] > states[1][2]:
			if observation[1] > states[1][2]:
				if observation[1] > states[1][0]:
					action = 0

				else:
					action = 1

		else:
			action = 1

	else:
		if observation[1] > states[1][0]:
			action = 0

		else:
			if observation[1] > states[1][2]:
				if observation[1] > states[1][0]:
					action = 0

				else:
					action = 1

	return action