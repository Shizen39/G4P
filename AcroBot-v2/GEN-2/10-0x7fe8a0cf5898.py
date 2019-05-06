def get_action(observation, all_obs):
	if observation[4] > all_obs[4][6]:
		if observation[0] <= all_obs[0][3]:
			if observation[2] <= all_obs[2][6]:
				if observation[0] <= all_obs[0][5]:
					action = 1

				else:
					action = 2

	else:
		if observation[5] > all_obs[5][8]:
			if observation[3] <= all_obs[3][1]:
				action = 2

			else:
				action = 2

		else:
			action = 2

			action = 0

		if observation[1] > all_obs[1][3]:
			action = 2

			action = 2

		else:
			action = 0

			action = 2

	return action