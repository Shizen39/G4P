def get_action(observation, states):
	if observation[3] > states[3][2]:
		if observation[1] <= states[1][0]:
			action = 1

			action = 0

			action = 1

			action = 1

		if observation[1] > states[1][2]:
			if observation[3] > states[3][2]:
				action = 0

			action = 1

			action = 1

		else:
			action = 1

			action = 0

			action = 1

			action = 1

		if observation[1] > states[1][2]:
			if observation[2] <= states[2][2]:
				if observation[2] <= states[2][2]:
					if observation[2] <= states[2][2]:
						action = 0

		else:
			if observation[2] <= states[2][2]:
				if observation[2] <= states[2][2]:
					if observation[2] <= states[2][2]:
						action = 0

		if observation[2] <= states[2][2]:
			if observation[2] <= states[2][2]:
				if observation[2] <= states[2][2]:
					if observation[2] <= states[2][2]:
						if observation[2] <= states[2][2]:
							action = 0

	if observation[2] <= states[2][2]:
		if observation[2] <= states[2][2]:
			if observation[2] <= states[2][2]:
				if observation[2] <= states[2][2]:
					if observation[2] <= states[2][2]:
						if observation[2] <= states[2][2]:
							if observation[2] <= states[2][2]:
								action = 0

	return action