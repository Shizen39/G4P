def get_action(observation, states):
	if observation[3] > states[3][2]:
		action = 1

	else:
		if observation[3] > states[3][2]:
			if observation[3] > states[3][2]:
				action = 1

			else:
				if observation[3] > states[3][2]:
					if observation[3] > states[3][2]:
						action = 1

					else:
						if observation[3] > states[3][2]:
							action = 1

						else:
							action = 1

				else:
					action = 1

		else:
			if observation[3] > states[3][2]:
				if observation[3] > states[3][2]:
					action = 1

				else:
					if observation[3] > states[3][2]:
						if observation[3] > states[3][2]:
							action = 1

						else:
							action = 1

					else:
						action = 0

			else:
				if observation[3] > states[3][2]:
					action = 1

				else:
					if observation[3] > states[3][2]:
						if observation[3] > states[3][2]:
							action = 1

						else:
							action = 1

					else:
						action = 0

	return action