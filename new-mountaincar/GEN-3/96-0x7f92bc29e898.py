def get_action(observation, all_obs):
	if observation[1] > all_obs[1][7]:
		if observation[0] <= all_obs[0][0]:
			if observation[0] <= all_obs[0][14]:
				if observation[0] > all_obs[0][3]:
					action = 2

					action = 1

				else:
					if observation[1] <= all_obs[1][2]:
						action = 2

					else:
						action = 0

			else:
				if observation[0] <= all_obs[0][2]:
					action = 1

					action = 2

				else:
					if observation[1] <= all_obs[1][2]:
						action = 1

		else:
			if observation[1] > all_obs[1][11]:
				if observation[0] <= all_obs[0][7]:
					if observation[1] <= all_obs[1][2]:
						action = 2

					else:
						action = 0

				else:
					if observation[0] <= all_obs[0][2]:
						action = 0

					else:
						action = 0

			else:
				if observation[1] <= all_obs[1][2]:
					action = 1

				if observation[1] > all_obs[1][11]:
					action = 2

				else:
					action = 1

	else:
		if observation[0] <= all_obs[0][4]:
			if observation[1] <= all_obs[1][6]:
				action = 0

			else:
				action = 2

		else:
			action = 1

			action = 0

	return action