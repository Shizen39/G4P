def get_action(observation, all_obs):
	if observation[0] <= all_obs[0][5]:
		if observation[1] > all_obs[1][3]:
			if observation[3] > all_obs[3][0]:
				action = 2

		else:
			action = 1

	if observation[4] <= all_obs[4][6]:
		action = 2

		action = 1

		if observation[4] > all_obs[4][6]:
			action = 1

		else:
			action = 2

	return action