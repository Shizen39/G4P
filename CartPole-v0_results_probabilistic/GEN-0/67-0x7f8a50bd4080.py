def get_action(observation, all_obs):
	if observation[2] > all_obs[2][2]:
		if observation[1] > all_obs[1][2]:
			action = 1

	if observation[1] <= all_obs[1][2]:
		if observation[3] > all_obs[3][3]:
			action = 1

	else:
		if observation[1] > all_obs[1][2]:
			action = 1

	return action