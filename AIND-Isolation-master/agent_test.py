"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent

from importlib import reload

from sample_players import (RandomPlayer, open_move_score,
                            improved_score, center_score)


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        #self.player1 = "Player1"
        #self.player2 = "Player2"
        self.player1 = game_agent.AlphaBetaPlayer()
        #self.player2 = game_agent.MinimaxPlayer()
        #self.player1 = game_agent.MinimaxPlayer(score_fn=improved_score)
        self.player2 = game_agent.MinimaxPlayer(score_fn=improved_score)
        self.game = isolation.Board(self.player1, self.player2)
    
    def test_prediction(self):
        self.game.apply_move((2, 3))
        self.game.apply_move((0, 5))
        #print(self.game.to_string())
        #assert(player1 == self.game.active_player)
    
        #print(self.game.get_legal_moves())
        #print(self.game.play(time_limit=600))
        
        #self.game.play(time_limit=1200)
        #self.game.history
        #print(self.game.history)
        #print(self.game.to_string())
        
        winner, history, outcome = self.game.play()
        print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
        print(self.game.to_string())
        print("Move history:\n{!s}".format(history))
        


if __name__ == '__main__':
    unittest.main()
