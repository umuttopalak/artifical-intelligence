from agent import Agent
from tictactoe import TicTacToe

game = TicTacToe()


agent = Agent(game, 'X', discount_factor= 0.6, episode=100000)

# agent.train_brain_o_byrandom()

agent.play_with_human()