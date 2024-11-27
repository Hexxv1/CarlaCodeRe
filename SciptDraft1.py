#!/usr/bin/env python3
import glob
import os
import sys
import random
import time
#import numpy as np
#import matplotlib.pyplot as plt
import carla

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_model(state_size, action_size):
    model = Sequential([
        Dense(24, input_dim=state_size, activation='relu'),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model(state_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_simulation(agent, vehicle):
    done = False
    state = np.zeros((1, agent.state_size))  # Initialize state
    total_reward = 0
    step = 0

    while not done:
        action = agent.act(state)
        # Convert action into throttle, steer, and brake
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=(action-1)*0.1))
        # Here you would need to compute the next state and the reward
        next_state = np.zeros((1, agent.state_size))
        reward = 1  # Define reward logic
        done = step > 100  # Define termination logic
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        step += 1

    return total_reward

def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 1985)
        client.set_timeout(2.0)
        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]  # Example vehicle
        spawn_point = random.choice(world.get_map().get_spawn_points())

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        state_size = 4
        action_size = 3
        agent = DQNAgent(state_size, action_size)

        episodes = 10
        results = []
        for episode in range(episodes):
            reward = run_simulation(agent, vehicle)
            results.append(f"Episode: {episode+1}, Reward: {reward}\n")
            print(f"Episode: {episode+1}, Reward: {reward}")
            agent.replay(32)

        # Save to a text file
        with open('simulation_results.txt', 'w') as file:
            file.writelines(results)

    finally:
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")

if __name__ == '__main__':
    main()
