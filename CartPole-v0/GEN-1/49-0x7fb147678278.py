def get_action(observation, all_obs):
	if observation[3] > all_obs[3][2]:
		if observation[2] <= all_obs[2][3]:
			if observation[3] > all_obs[3][3]:
				action = 1

		else:
			if observation[2] > all_obs[2][1]:
				action = 1

	else:
		action = 0

	return action