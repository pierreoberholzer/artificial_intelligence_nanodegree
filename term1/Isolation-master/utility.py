def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

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

    #if game.is_loser(player):
    #   return float("-inf")

    #if game.is_winner(player):
    #   return float("inf")
    
    
    def find_fronteer(tuple_ls,width,height):
        tuple_ls.sort()
        fronteer = []
        for b in tuple_ls:
            r,c = b[0],b[1]
            if r != 0 and c!= 0:
                next
            elif (r == 0 or c == 0) and (len(fronteer) == 0): # We start from a border
                r_start,c_start = r,c
                fronteer.append((r_start,c_start))
                r_last,c_last = fronteer[-1][0],fronteer[-1][1]
            elif (r == r_last and (c == c_last+1 or c == c_last-1)) or (c == c_last and (r == r_last+1 or r == r_last-1)):
                fronteer.append((r,c))
                r_last,c_last = fronteer[-1][0],fronteer[-1][1]
            elif (r == 0 or r == height or c == 0 or c == width):
                break
            else:
                fronteer = []
        if len(fronteer) >= 3:
            print("Fronteer found:")
            print(fronteer)
            return fronteer
        else:
            return None
    
    
    if game.is_loser(player):
       return float("-inf")

    if game.is_winner(player):
       return float("inf")
    
    full_board = [(i,j) for i in range(game.width) for j in range(game.height)]
    blanks = game.get_blank_spaces()
    not_blanks = [i for i in full_board if i not in blanks]
    
    fronteer = find_fronteer(not_blanks,game.width,game.height)
    
    #own_moves = game.get_legal_moves(player)
    #opp_moves = game.get_legal_moves(game.get_opponent(player))
    
    #print(blanks)
    
    #import pdb;pdb.set_trace()
    
    if fronteer:
        
        #print("length of fronteer is:")
        #import pdb;pdb.set_trace()
        #print(len(fronteer))
        #return float("-inf")
        #print(fronteer)
        return(len(fronteer))
    else:
        
        #print("no fronteer is")
        #return float(len(fronteer))
        #return float("inf")
        return 0.