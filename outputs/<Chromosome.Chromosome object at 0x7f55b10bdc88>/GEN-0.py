def get_action(observation, states):
	if observation[3] <= states[3][0]:
		if observation[2] > states[2][2]:
			action = 0

	else:
		if observation[3] <= states[3][0]:
			if observation[2] > states[2][2]:
				action = 0

		else:
			if observation[3] <= states[3][0]:
				if observation[2] > states[2][2]:
					action = 0

			else:
				if observation[3] <= states[3][0]:
					if observation[2] > states[2][2]:
						action = 0

				else:
					action = 1

	return action