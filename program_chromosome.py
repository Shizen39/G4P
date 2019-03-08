def get_action(observation, states):
	if observation[2] > states[2][0]:
    		if observation[2] > states[2][1]:
			if observation[3] > states[3][0]:
				if observation[2] <= states[2][0]:
					action = 1

		else:
			if observation[3] > states[3][0]:
				if observation[2] <= states[2][0]:
					action = 1

	else:
		if observation[3] > states[3][0]:
			if observation[2] <= states[2][0]:
				if observation[3] <= states[3][2]:
					action = 0

	return action