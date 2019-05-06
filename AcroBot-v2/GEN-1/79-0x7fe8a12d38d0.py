def get_action(observation, all_obs):
	if observation[0] <= all_obs[0][5]:
		action = 2

	if observation[4] <= all_obs[4][6]:
		if observation[0] <= all_obs[0][5]:
			action = 1

			action = 2

		else:
			action = 2

	return action