def get_action(observation, all_obs):
	if observation[2] <= all_obs[2][5]:
		if observation[2] <= all_obs[2][3]:
			action = 0

		if observation[2] > all_obs[2][3]:
			action = 1

	else:
		if observation[2] <= all_obs[2][1]:
			if observation[3] > all_obs[3][3]:
				action = 1

	return action