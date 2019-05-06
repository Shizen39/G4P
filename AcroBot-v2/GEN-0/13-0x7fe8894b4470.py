def get_action(observation, all_obs):
	if observation[4] <= all_obs[4][6]:
		if observation[0] <= all_obs[0][0]:
			action = 0

		else:
			action = 1

			action = 1

			action = 1

			action = 1

	else:
		if observation[0] <= all_obs[0][0]:
			action = 2

		else:
			action = 0

	return action