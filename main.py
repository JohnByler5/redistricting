import os
import datetime as dt

from stable_baselines3 import PPO

from redistricting_env import RedistrictingEnv


def time(start):
    return f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))}'


def main():
    start = dt.datetime.now()

    print(f'{time(start)} - Creating env...')

    action_polygon_points = 4
    verbose = True
    weights = {}

    env = RedistrictingEnv('data/pa/WP_VotingDistricts.shp', action_polygon_points=action_polygon_points,
                           verbose=verbose, start=start, weights=weights)

    # Create the PPO agent
    print(f'{time(start)} - Creating agent...')
    policy_kwargs = {
        'net_arch': [
            int(env.n_districts * env.action_polygon_points / 1),
            int(env.n_districts * env.action_polygon_points / 2),
            int(env.n_districts * env.action_polygon_points / 4),
            int(env.n_districts * env.action_polygon_points / 8),
        ]
    }
    model = PPO('MlpPolicy', env, n_steps=env.n_districts, batch_size=env.n_districts, policy_kwargs=policy_kwargs,
                verbose=0)

    # Train the agent
    simulations = 1000
    timesteps_per_simulation = env.n_districts
    print(f'{time(start)} - Beginning agent training for {simulations} simulations...')
    model.learn(total_timesteps=simulations * timesteps_per_simulation)

    # Save the agent
    print(f'{time(start)} - Saving agent...')
    count = len(os.listdir('models'))
    model.save(f'models/model_{count}')


if __name__ == '__main__':
    main()
