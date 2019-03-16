def get_action(observation, all_obs):
	if observation[2] <= all_obs[2][3]:
		if observation[0] <= all_obs[0][3]:
			action = 1

		if observation[0] > all_obs[0][4]:
			action = 0

	else:
		if observation[1] > all_obs[1][1]:
			if observation[3] > all_obs[3][2]:
				action = 1

		else:
			if observation[3] > all_obs[3][3]:
				action = 1

			else:
				action = 0

	return action