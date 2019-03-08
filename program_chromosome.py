def get_action(observation, states):
	if observation[2] > states[2][1]:
		action = 0

	else:
		if observation[2] > states[2][2]:
			if observation[2] > states[2][1]:
				action = 0

			else:
				if observation[2] > states[2][2]:
					action = 0

				else:
					action = 0

		else:
			action = 0

	return action