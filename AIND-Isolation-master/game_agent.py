"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
#from random import randint
import numpy as np

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    
    This heuristic calculates the number of legal moves available to the current
    player that are not available to the opponent player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
       return float("-inf")

    if game.is_winner(player):
       return float("inf")
   
    # List of moves available for each player
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    
    # List of moves available for each player and not available to the other player
    only_own_moves = [c for c in own_moves if c not in opp_moves ]
    only_opp_moves = [c for c in opp_moves if c not in own_moves ]
    
    return float(len(only_own_moves) - len(only_opp_moves))

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic returns a score that is inversely proportional to the distance
    of the current player to the centroid of the free area (i.e. made of the blank
    boxes on the board game).

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    def centeroidnp(arr):
        """Calculate the centroid of an array.
        
        Parameters
        ----------
        arr: numpy array of the boxes, i.e. [[0,1],[4,3],[2,1]]
    
        Returns
        -------   
        float,float
            Position of the centroid
        
        Note
        ----
        - Reference given in report "heuristic_analysis.pdf"
        - Function reported to be faster than arr.mean(arr, axis = 0)
        """
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length
    
    blanks_array = np.array(game.get_blank_spaces())
        
    centroid = centeroidnp(blanks_array)

    center_h, center_w = centroid[0],centroid[1]
          
    y, x = game.get_player_location(player)
    
    dist  = float((center_h - x)**2 + (center_w - y)**2)

    return(1/(dist+0.0001))
     

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic returns a score depending on a threshold reprenting the 
    progress in the game, either the score of custom_score_2 (see below) or 
    of center_score (see sample_player.py).

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    board_size = game.width*game.height # Used to calculate progress in the game
    blanks_threshold = 0.5 # Threshold used to change scoring method

    blanks_array = np.array(game.get_blank_spaces())

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    def centeroidnp(arr):
        """Calculate the centroid of an array.
    
        Parameters
        ----------
        arr: numpy array of the boxes, i.e. [[0,1],[4,3],[2,1]]
    
        Returns
        -------   
        float,float
            Position of the centroid
        
        Note
        ----
        - Reference given in report "heuristic_analysis.pdf"
        - Function reported to be faster than arr.mean(arr, axis = 0)
        """
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length
    
    centroid = centeroidnp(blanks_array)

    w, h = game.width / 2., game.height / 2.
    center_h, center_w = centroid[0],centroid[1]
          
    y, x = game.get_player_location(player)

    if len(blanks_array) < blanks_threshold*board_size:
        dist  = float((h - y)**2 + (w - x)**2)
        return(1/(dist+0.0001))    
    else:
        dist  = float((h - y)**2 + (w - x)**2)
        return(dist)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move =  self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        gameState = game
        

        def minimax_decision(self,gameState):
            """ Return the move along a branch of the game tree that
            has the best possible value.  A move is a pair of coordinates
            in (column, row) order corresponding to a legal move for
            the searching player.
    
            You can ignore the special case of calling this function
            from a terminal state.
            """
            curdepth = 1
            
            return max(gameState.get_legal_moves(),
               key=lambda m: min_value(self,gameState.forecast_move(m),curdepth))

        def terminal_test(gameState):
            """ Return True if the game is over for the active player
            and False otherwise.
            """
            return not bool(gameState.get_legal_moves())  # by Assumption 1

        def min_value(self,gameState,curdepth):
            """ Return the value for the heuristic function if the game is over
            or the maximal depth reached,otherwise return the minimum value over
            all legal child nodes.
            """
                  
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if terminal_test(gameState) or curdepth == self.search_depth:

                return self.score(gameState,gameState._inactive_player)
                       
            v = float("inf")
            
            curdepth += 1
            
            for m in gameState.get_legal_moves():

                v = min(v, max_value(self,gameState.forecast_move(m),curdepth))
            
            return v
                        
        def max_value(self,gameState,curdepth):
            """ Return the value for the heuristic function if the game is over
            or the maximal depth reached,otherwise return the maximum value over
            all legal child nodes.
            """
            
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if terminal_test(gameState) or curdepth == self.search_depth:

                return self.score(gameState,gameState._active_player)
                
            v = float("-inf")            
            
            curdepth += 1
            
            for m in gameState.get_legal_moves():
                
                v = max(v, min_value(self,gameState.forecast_move(m),curdepth))
            
            return v
            
        if not gameState.get_legal_moves():
            return (-1,-1)

        return minimax_decision(self,gameState)

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        
        self.search_depth = 1

        # Iterative deepening as in:
        # https://github.com/aimacode/aima-pseudocode/blob/master/md/Iterative-Deepening-Search.md
        
        try:
            while True:
                
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_move =  self.alphabeta(game, self.search_depth) 
                self.search_depth += 1
    
        except SearchTimeout:
            pass
            #return best_move # Handle any actions required after timeout as needed

        # In case best_move returned not in legal moves, randomly pick up one
        legal_moves = game.get_legal_moves()
        
        if legal_moves and best_move not in legal_moves:
            best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        gameState = game
        
                 
        if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
        
        def alpha_beta_search(self,gameState,alpha,beta):
            """ Return the move along a branch of the game tree that
            has the best possible value.  A move is a pair of coordinates
            in (column, row) order corresponding to a legal move for
            the searching player.
    
            You can ignore the special case of calling this function
            from a terminal state.
            """
            
            # Algorithm presented in:
            # https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
            
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            curdepth = 1

            best_score = float("-inf") 
            best_move = None

            for m in gameState.get_legal_moves():
                
                v = min_value(self,gameState.forecast_move(m),curdepth,alpha,beta)
                alpha = max(alpha,v)
                if v > best_score:
                    best_score = v
                    best_move = m
                    
            return best_move


        def terminal_test(gameState):
            """ Return True if the game is over for the active player
            and False otherwise.
            """
            
            # Algorithm presented in:
            # https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
                
            return not bool(gameState.get_legal_moves())  # by Assumption 1

        def min_value(self,gameState,curdepth,alpha,beta):
            """ Return the value for the heuristic function if the game is over
            or the maximal depth reached,otherwise return the minimum value over
            all legal child nodes.
            """
            
            # Algorithm presented in:
            # https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
                  
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if terminal_test(gameState) or curdepth == self.search_depth:

                return self.score(gameState,gameState._inactive_player)
                       
            v = float("inf")
            
            curdepth += 1
            
            for m in gameState.get_legal_moves():
                
                v = min(v, max_value(self,gameState.forecast_move(m),curdepth,alpha,beta))
                if v <= alpha: 
                    return v
                beta = min(beta,v)
            
            return v
                        

        def max_value(self,gameState,curdepth,alpha,beta):
            """ Return the value for the heuristic function if the game is over
            or the maximal depth reached,otherwise return the maximum value over
            all legal child nodes.
            """
            
            # Algorithm presented in:
            # https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
            
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if terminal_test(gameState) or curdepth == self.search_depth:
                
                return self.score(gameState,gameState._active_player)
                
            v = float("-inf")            
            
            curdepth += 1
            
            for m in gameState.get_legal_moves():
                
                v = max(v, min_value(self,gameState.forecast_move(m),curdepth,alpha,beta))
                if v >= beta: 
                    return v
                alpha = max(alpha,v)
            
            return v

        if not gameState.get_legal_moves():
            return (-1,-1)

        return alpha_beta_search(self,gameState,alpha,beta)


    