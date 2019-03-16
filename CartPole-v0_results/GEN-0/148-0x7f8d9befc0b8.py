def get_action(observation, states):
	if observation[2] <= states[2][5]:
		if observation[2] <= states[2][3]:
			action = 0

		if observation[2] > states[2][3]:
			action = 1

	else:
		if observation[2] <= states[2][1]:
			if observation[3] > states[3][3]:
				action = 1

	return action