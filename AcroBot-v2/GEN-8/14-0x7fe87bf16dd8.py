def get_action(observation, all_obs):
	if observation[0] <= all_obs[0][3]:
		if observation[0] <= all_obs[0][0]:
			if observation[5] > all_obs[5][5]:
				action = 0

			else:
				action = 2

		else:
			if observation[2] > all_obs[2][11]:
				action = 2

	else:
		if observation[4] <= all_obs[4][6]:
			action = 2

			action = 1

			if observation[5] <= all_obs[5][2]:
				action = 2

			else:
				action = 2

	return action