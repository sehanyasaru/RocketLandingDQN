import pygame
from rocket_landing_env import Rocket, DQNAgent, background_img, screen


def run_simulation():
    rocket = Rocket()
    agent = DQNAgent()
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = rocket.get_state()
        action = agent.select_action(state)
        reward = rocket.update(action)

        total_reward = -0.1
        total_reward += rocket.apply_rotation_penalty()
        total_reward += rocket.stabilize_rotation_reward()


        if rocket.vy > 0:
            total_reward += 1
        elif rocket.vy < 0:
            total_reward -= 0.5

        agent.store_experience(state, action, total_reward + reward, rocket.get_state(), rocket.landed or rocket.crashed)
        agent.train()

        screen.blit(background_img, (0, 0))
        rocket.draw()
        pygame.display.flip()
        clock.tick(30)

        if rocket.landed or rocket.crashed:
            print(f"Episode Reward: {total_reward + reward}")
            rocket.reset()

    pygame.quit()


run_simulation()
