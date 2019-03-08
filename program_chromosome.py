def get_action(observation, states):
	if observation[3] <= states[3][1]:
		if observation[0] <= states[0][1]:
			if observation[1] <= states[1][1]:
				if observation[3] <= states[3][1]:
					action = 0

				else:
					action = 0

		else:
			if observation[1] <= states[1][1]:
				if observation[3] <= states[3][1]:
					action = 0

				else:
					action = 0

	else:
		if observation[1] <= states[1][1]:
			if observation[3] <= states[3][1]:
				if observation[0] <= states[0][1]:
					action = 1

				else:
					action = 1

			else:
				if observation[3] <= states[3][1]:
					action = 0

				else:
					action = 0

	return action