def get_action(observation, states):
	if observation[1] > states[1][2]:
		if observation[3] > states[3][2]:
			action = 1

	else:
		if observation[1] > states[1][2]:
			if observation[3] > states[3][2]:
				action = 1

		else:
			if observation[1] > states[1][2]:
				if observation[3] > states[3][2]:
					action = 1

			else:
				if observation[1] > states[1][2]:
					if observation[3] > states[3][2]:
						action = 1

				else:
					action = 1

	return action