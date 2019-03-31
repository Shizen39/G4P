def get_action(observation, all_obs):
	if observation[1] <= all_obs[1][4]:
		if observation[0] > all_obs[0][1]:
			action = 0

		else:
			action = 1

		if observation[1] > all_obs[1][6]:
			action = 1

		if observation[1] <= all_obs[1][4]:
			action = 1

		else:
			action = 2

		if observation[1] <= all_obs[1][5]:
			action = 0

		else:
			action = 2

	else:
		if observation[1] <= all_obs[1][4]:
			action = 2

			if observation[0] <= all_obs[0][7]:
				action = 1

		else:
			if observation[0] <= all_obs[0][3]:
				action = 2

			if observation[1] <= all_obs[1][5]:
				action = 0

			else:
				action = 2

	return action