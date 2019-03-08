def get_action(observation, states):
	if observation[2] <= states[2][1]:
		if observation[3] <= states[3][2]:
			if observation[0] > states[0][1]:
				if observation[2] <= states[2][0]:
					action = 0

				else:
					action = 0

			else:
				if observation[2] <= states[2][1]:
					action = 1

				else:
					action = 0

		else:
			if observation[0] > states[0][1]:
				if observation[2] <= states[2][0]:
					action = 0

				else:
					action = 0

			else:
				if observation[2] <= states[2][1]:
					action = 1

				else:
					action = 0

	else:
		if observation[0] > states[0][1]:
			if observation[2] <= states[2][0]:
				if observation[0] <= states[0][2]:
					action = 0

				else:
					action = 0

			else:
				if observation[3] <= states[3][2]:
					action = 0

				else:
					action = 0

		else:
			if observation[2] <= states[2][0]:
				if observation[0] <= states[0][2]:
					action = 0

				else:
					action = 0

			else:
				if observation[3] <= states[3][2]:
					action = 0

				else:
					action = 0

	return action