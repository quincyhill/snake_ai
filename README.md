# AI Based Snake Game using Reinformcent Learning via Q-Learning

Organization

## Agent

- Game
- Model

### Training

- state = get_state(game)
- action = get_move(state)
  - model.predict()
- reward, game_over, score = game.play_step(action)
- new_state = get_state(game)
- remember(state, action, reward, new_state, game_over)
- model.train()

## Game

- play_step(action)
  -> reward, game_over, score

## Model

Linear_QNet(DQN)

- model.predict(state)
  -> action

Reward

- eat food: +10
- game over: -10
- else: 0

Action (3 values)

[1, 0, 0] -> straight
[0, 1, 0] -> right turn
[0, 0, 1] -> left turn

State (14 values)

[danger wall straight, danger wall right, danger wall left, danger snake straight, danger snake right, danger snake left, direction left, direction right, direction up, direction down, food left, food right, food up, food down]
