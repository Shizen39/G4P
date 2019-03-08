def get_action(observation, states):
	if observation[2] <= states[2][2]:
		if observation[2] > states[2][1]:
			if observation[1] > states[1][1]:
				if observation[2] <= states[2][2]:
					action = 1

			else:
				if observation[1] > states[1][1]:
					action = 0

				else:
					action = 0

		else:
			if observation[1] <= states[1][1]:
				if observation[2] <= states[2][2]:
					action = 0

				else:
					action = 1

	else:
		if observation[1] > states[1][1]:
			if observation[2] <= states[2][2]:
				if observation[1] <= states[1][1]:
					action = 0

		else:
			if observation[2] <= states[2][2]:
				if observation[1] <= states[1][1]:
					action = 0

	return action