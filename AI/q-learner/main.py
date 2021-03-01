import model as qm


if __name__ == "__main__":

    ENV_OPTIONS = ["FrozenLake-v0", "Taxi-v3"]
    ENV_VICTORY_REWARDS = [1, 20]

    print("Welcome to Q-Learner")
    print("A simple experimental lab for small reinforcement learning models\n\n")

    while True:
        opt = str(input("Do you want to train and test a model (Y/n)? "))
        if opt == 'n' or opt == 'N':
            print("Bye bye!")
            break

        # Get Envirionment Input Option
        print("Plase select a game envrionment to train your model")
        print("Available options:")
        for i, name in zip(range(1,len(ENV_OPTIONS)+1), ENV_OPTIONS):
            print("%d. %s" % (i, name))

        envOpt = int(input("Option: "))
        if envOpt <= 0 or envOpt > len(ENV_OPTIONS):
            print("Invalid Option Entered!")
            continue
        selectedEnv = ENV_OPTIONS[envOpt - 1]

        # Get Hyper Parameters
        n_episodes = int(input("N Episodes (8000): "))
        max_steps_per_episode = int(input("Max steps per episode (100): "))

        learning_rate = float(input("Learning Rate (0.1): "))
        discount_rate = float(input("Discounting Rate (0.99): "))

        min_exploration_rate = float(input("Exploration Rate Minimum (0.002): "))
        exploration_decay_rate = float(input("Exploration Rate Decay (0.003): "))

        # Train Model
        model, world = qm.initModel(selectedEnv)

        qm.trainModel(model, world, n_episodes, max_steps_per_episode, learning_rate, discount_rate, min_exploration_rate, exploration_decay_rate)
        # Show Scores

        # Demo Model
        opt = str(input("Do you want to see the model live running (y/N)?: "))
        if opt == 'y' or opt == 'Y':
            qm.playModel(model, world, max_steps_per_episode, ENV_VICTORY_REWARDS[envOpt - 1])


