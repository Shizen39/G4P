def get_action(observation, all_obs):
	if observation[1] > all_obs[1][7]:
		if observation[1] <= all_obs[1][3]:
			if observation[1] <= all_obs[1][0]:
				action = 1

		else:
			action = 1

			action = 2

	else:
		if observation[0] > all_obs[0][5]:
			if observation[0] <= all_obs[0][3]:
				if observation[0] <= all_obs[0][0]:
					if observation[1] <= all_obs[1][8]:
						if observation[1] > all_obs[1][8]:
							if observation[1] <= all_obs[1][8]:
								if observation[0] > all_obs[0][4]:
									action = 0

				else:
					if observation[0] > all_obs[0][5]:
						action = 0

					else:
						action = 0

					if observation[0] <= all_obs[0][4]:
						action = 0

					else:
						action = 2

					if observation[1] > all_obs[1][8]:
						if observation[1] <= all_obs[1][8]:
							action = 1

					if observation[0] > all_obs[0][9]:
						action = 1

						action = 1

						action = 0

						action = 0

					else:
						if observation[0] <= all_obs[0][4]:
							if observation[1] <= all_obs[1][8]:
								action = 1

						else:
							if observation[0] <= all_obs[0][13]:
								action = 0

							else:
								action = 1

		else:
			if observation[0] > all_obs[0][9]:
				if observation[0] > all_obs[0][5]:
					if observation[0] <= all_obs[0][3]:
						action = 0

				else:
					if observation[0] <= all_obs[0][15]:
						action = 2

					else:
						action = 1

				if observation[0] <= all_obs[0][13]:
					if observation[0] > all_obs[0][4]:
						action = 1

				else:
					if observation[1] > all_obs[1][2]:
						action = 2

					else:
						action = 1

				if observation[1] <= all_obs[1][8]:
					if observation[1] <= all_obs[1][8]:
						action = 0

					if observation[0] <= all_obs[0][12]:
						action = 1

				else:
					if observation[0] > all_obs[0][0]:
						if observation[1] <= all_obs[1][5]:
							action = 1

					else:
						if observation[1] <= all_obs[1][4]:
							action = 1

				if observation[1] <= all_obs[1][8]:
					if observation[1] <= all_obs[1][8]:
						if observation[0] > all_obs[0][0]:
							action = 0

						else:
							action = 1

					if observation[0] > all_obs[0][0]:
						if observation[1] <= all_obs[1][5]:
							action = 1

					else:
						if observation[1] <= all_obs[1][4]:
							action = 1

				else:
					if observation[1] <= all_obs[1][8]:
						if observation[1] <= all_obs[1][8]:
							action = 0

						if observation[0] <= all_obs[0][12]:
							action = 1

					else:
						if observation[0] > all_obs[0][0]:
							if observation[1] <= all_obs[1][5]:
								action = 1

						else:
							if observation[1] <= all_obs[1][4]:
								action = 1

			else:
				if observation[1] <= all_obs[1][8]:
					if observation[1] <= all_obs[1][8]:
						if observation[0] > all_obs[0][0]:
							action = 1

							action = 1

						else:
							if observation[1] <= all_obs[1][8]:
								action = 1

					if observation[0] > all_obs[0][9]:
						action = 2

						action = 1

						if observation[1] <= all_obs[1][8]:
							action = 0

						else:
							action = 0

					else:
						if observation[0] <= all_obs[0][15]:
							if observation[0] <= all_obs[0][13]:
								action = 1

						else:
							if observation[1] <= all_obs[1][8]:
								action = 1

				else:
					if observation[0] > all_obs[0][9]:
						action = 1

						action = 2

						if observation[0] <= all_obs[0][3]:
							action = 0

						if observation[0] <= all_obs[0][15]:
							if observation[0] <= all_obs[0][13]:
								action = 1

						else:
							if observation[1] <= all_obs[1][8]:
								action = 1

					else:
						if observation[0] > all_obs[0][9]:
							action = 1

							action = 1

							action = 0

							action = 0

						else:
							if observation[0] <= all_obs[0][4]:
								if observation[1] <= all_obs[1][8]:
									action = 1

							else:
								if observation[0] <= all_obs[0][13]:
									action = 0

								else:
									action = 1

	return action