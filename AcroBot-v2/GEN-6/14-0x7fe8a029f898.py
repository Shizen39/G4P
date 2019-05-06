def get_action(observation, all_obs):
	if observation[0] <= all_obs[0][3]:
		if observation[4] > all_obs[4][7]:
			action = 0

		else:
			action = 1

			action = 1

			action = 2

	else:
		if observation[4] <= all_obs[4][6]:
			action = 0

			if observation[4] > all_obs[4][6]:
				action = 0

			else:
				action = 2

	return action