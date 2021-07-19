import datetime
import os

import numpy
import itertools
import copy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (1,8,8)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(4672))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 512  # Maximum number of moves if game is not finished before
        self.num_simulations = 700  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD //// Adam has more fast convergence and SGD is better but with longer training time
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Chess()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.
        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.
        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
    
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (1 to 8) to play for the player {self.to_play()}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1 to 8) to play for the player {self.to_play()}: "
                    )
                )
                choice = (row - 1) * 8 + (col - 1)
                if (
                    choice in self.legal_actions()):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        row = action_number // 8 + 1
        col = action_number % 8 + 1
        return f"Play row {row}, column {col}"


class Chess:
    def __init__(self):
        #self.board = numpy.zeros((8, 8), dtype="int32")
        self.board = numpy.zeros([8,8]).astype(str)
        self.board[0,0] = 1 #"r"
        self.board[0,1] = 2 #"n"
        self.board[0,2] = 3 #"b"
        self.board[0,3] = 4 #"q"
        self.board[0,4] = 5 #"k"
        self.board[0,5] = 3 #"b"
        self.board[0,6] = 2 #"n"
        self.board[0,7] = 1 #"r"
        self.board[1,0:8] = 6 #"p"
        self.board[7,0] = 11 #"R"
        self.board[7,1] = 12 #"N"
        self.board[7,2] = 13 #"B"
        self.board[7,3] = 14 #"Q"
        self.board[7,4] = 15 #"K"
        self.board[7,5] = 13 #"B"
        self.board[7,6] = 12 #"N"
        self.board[7,7] = 11 #"R"
        self.board[6,0:8] = 16 #"P"
        #self.board[self.board] = 0
        self.move_count = 0
        self.no_progress_count = 0
        self.repetitions_w = 0
        self.repetitions_b = 0
        self.move_history = None
        self.en_passant = -999; self.en_passant_move = 0 # returns j index of last en_passant pawn
        self.r1_move_count = 0 # black's queenside rook
        self.r2_move_count = 0 # black's kingside rook
        self.k_move_count = 0
        self.R1_move_count = 0 # white's queenside rook
        self.R2_move_count = 0 # white's kingside rook
        self.K_move_count = 0
        self.current_board = self.board
        self.en_passant_move_copy = None
        self.copy_board = None; self.en_passant_copy = None; self.r1_move_count_copy = None; self.r2_move_count_copy = None; 
        self.k_move_count_copy = None; self.R1_move_count_copy = None; self.R2_move_count_copy = None; self.K_move_count_copy = None
        #self.player = 1 # current player's turn (0:white, 1:black)

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((8, 8), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        row = action // 8
        col = action % 8
        self.board[row, col] = self.player


        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        '''board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == -1, 1, 0)'''
        board_to_play = numpy.full((8, 8), self.player)
        return numpy.array([board_to_play], dtype="int32")


    def legal_actions(self): # returns all possible actions while not in check: initial_pos,final_pos,underpromote
        acts = []
        if self.player == 0:
            _,c_dict = self.possible_W_moves() # all non-king moves except castling
            current_position = numpy.where(self.current_board==15)
            i, j = current_position; i,j = i[0],j[0]
            c_dict[15] = {(i,j):self.move_rules_K()} # all king moves
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        if key in [16,6] and final_pos[0] in [0,7]:
                            for p in ["queen","rook","knight","bishop"]:
                                acts.append([initial_pos,final_pos,p])
                        else:
                            acts.append([initial_pos,final_pos,None])
            actss = []
            for act in acts:  ## after move, check that its not check ownself, else illegal move
                i,f,p = act; b = copy.deepcopy(self)
                b.move_piece(i,f,p)
                b.player = 0
                if b.check_status() == False:
                    actss.append(act)
            return actss
        if self.player == 1:
            _,c_dict = self.possible_B_moves() # all non-king moves except castling
            current_position = numpy.where(self.current_board==5)
            i, j = current_position; i,j = i[0],j[0]
            c_dict[5] = {(i,j):self.move_rules_k()} # all king moves
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        if key in [16,6] and final_pos[0] in [0,7]:
                            for p in ["queen","rook","knight","bishop"]:
                                acts.append([initial_pos,final_pos,p])
                        else:
                            acts.append([initial_pos,final_pos,None])
            actss = []
            for act in acts:  ## after move, check that its not check ownself, else illegal move
                i,f,p = act; b = copy.deepcopy(self)
                b.move_piece(i,f,p)
                b.player = 1
                if b.check_status() == False:
                    actss.append(act)
            return actss


        #does not include king, castling
    def possible_W_moves(self, threats=False):
        board_state = self.current_board[i,j]
        rooks = {}; knights = {}; bishops = {}; queens = {}; pawns = {};
        i,j = numpy.where(board_state==11)
        for rook in zip(i,j):
            rooks[tuple(rook)] = self.move_rules_R(rook)
        i,j = numpy.where(board_state==12)
        for knight in zip(i,j):
            knights[tuple(knight)] = self.move_rules_N(knight)
        i,j = numpy.where(board_state==13)
        for bishop in zip(i,j):
            bishops[tuple(bishop)] = self.move_rules_B(bishop)
        i,j = numpy.where(board_state==14)
        for queen in zip(i,j):
            queens[tuple(queen)] = self.move_rules_Q(queen)
        i,j = numpy.where(board_state==16)
        for pawn in zip(i,j):
            if threats==False:
                pawns[tuple(pawn)],_ = self.move_rules_P(pawn)
            else:
                _,pawns[tuple(pawn)] = self.move_rules_P(pawn)
        c_dict = {11: rooks, 12: knights, 13: bishops, 14: queens, 16: pawns}
        c_list = []
        c_list.extend(list(itertools.chain(*list(rooks.values())))); c_list.extend(list(itertools.chain(*list(knights.values())))); 
        c_list.extend(list(itertools.chain(*list(bishops.values())))); c_list.extend(list(itertools.chain(*list(queens.values()))))
        c_list.extend(list(itertools.chain(*list(pawns.values()))))
        return c_list, c_dict
        
    def move_rules_k(self):
        current_position = numpy.where(self.current_board==15)
        i, j = current_position; i,j = i[0],j[0]
        next_positions = []
        c_list, _ = self.possible_W_moves(threats=True)
        for a,b in [(i+1,j),(i-1,j),(i,j+1),(i,j-1),(i+1,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1)]:
            if 0<=a<=7 and 0<=b<=7:
                if self.current_board[a,b] in [0,14,13,12,16,11] and (a,b) not in c_list:
                    next_positions.append((a,b))
        if self.castle("queenside") == True and self.check_status() == False:
            next_positions.append((0,2))
        if self.castle("kingside") == True and self.check_status() == False:
            next_positions.append((0,6))
        return next_positions
    
        #does not include king, castling
    def possible_B_moves(self,threats=False):
        rooks = {}; knights = {}; bishops = {}; queens = {}; pawns = {};
        board_state = self.current_board
        i,j = numpy.where(board_state==1)
        for rook in zip(i,j):
            rooks[tuple(rook)] = self.move_rules_r(rook)
        i,j = numpy.where(board_state==2)
        for knight in zip(i,j):
            knights[tuple(knight)] = self.move_rules_n(knight)
        i,j = numpy.where(board_state==3)
        for bishop in zip(i,j):
            bishops[tuple(bishop)] = self.move_rules_b(bishop)
        i,j = numpy.where(board_state==4)
        for queen in zip(i,j):
            queens[tuple(queen)] = self.move_rules_q(queen)
        i,j = numpy.where(board_state==6)
        for pawn in zip(i,j):
            if threats==False:
                pawns[tuple(pawn)],_ = self.move_rules_p(pawn)
            else:
                _,pawns[tuple(pawn)] = self.move_rules_p(pawn)
        c_dict = {1: rooks, 2: knights, 3: bishops, 4: queens, 6: pawns}
        c_list = []
        c_list.extend(list(itertools.chain(*list(rooks.values())))); c_list.extend(list(itertools.chain(*list(knights.values())))); 
        c_list.extend(list(itertools.chain(*list(bishops.values())))); c_list.extend(list(itertools.chain(*list(queens.values()))))
        c_list.extend(list(itertools.chain(*list(pawns.values()))))
        return c_list, c_dict
        
    def move_rules_K(self):
        current_position = numpy.where(self.current_board==15)
        i, j = current_position; i,j = i[0],j[0]
        next_positions = []
        c_list, _ = self.possible_B_moves(threats=True)
        for a,b in [(i+1,j),(i-1,j),(i,j+1),(i,j-1),(i+1,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1)]:
            if 0<=a<=7 and 0<=b<=7:
                if self.current_board[a,b] in [0,4,3,2,6,1] and (a,b) not in c_list:
                    next_positions.append((a,b))
        if self.castle("queenside") == True and self.check_status() == False:
            next_positions.append((7,2))
        if self.castle("kingside") == True and self.check_status() == False:
            next_positions.append((7,6))
        return next_positions
    
    def move_piece(self,initial_position,final_position,promoted_piece=14):
        if self.player == 0:
            promoted = False
            i, j = initial_position
            piece = self.current_board[i,j]
            self.current_board[i,j] = 0
            i, j = final_position
            if piece == 11 and initial_position == (7,0):
                self.R1_move_count += 1
            if piece == 11 and initial_position == (7,7):
                self.R2_move_count += 1
            if piece == 15:
                self.K_move_count += 1
            x, y = initial_position
            if piece == 16:
                if abs(x-i) > 1:
                    self.en_passant = j; self.en_passant_move = self.move_count
                if abs(y-j) == 1 and self.current_board[i,j] == 0: # En passant capture
                    self.current_board[i+1,j] = 0
                if i == 0 and promoted_piece in [11,13,12,14]:
                    self.current_board[i,j] = promoted_piece
                    promoted = True
            if promoted == False:
                self.current_board[i,j] = piece
            self.player = 1
            self.move_count += 1
    
        elif self.player == 1:
            promoted = False
            i, j = initial_position
            piece = self.current_board[i,j]
            self.current_board[i,j] = 0
            i, j = final_position
            if piece == 1 and initial_position == (0,0):
                self.r1_move_count += 1
            if piece == 1 and initial_position == (0,7):
                self.r2_move_count += 1
            if piece == 5:
                self.k_move_count += 1
            x, y = initial_position
            if piece == 6:
                if abs(x-i) > 1:
                    self.en_passant = j; self.en_passant_move = self.move_count
                if abs(y-j) == 1 and self.current_board[i,j] == 0: # En passant capture
                    self.current_board[i-1,j] = 0
                if i == 7 and promoted_piece in [1,3,2,4]:
                    self.current_board[i,j] = promoted_piece
                    promoted = True
            if promoted == False:
                self.current_board[i,j] = piece
            self.player = 0
            self.move_count += 1

        else:
            print("Invalid move: ",initial_position,final_position,promoted_piece)


    ## Check if current player's king is in check
    def check_status(self):
        if self.player == 0:
            c_list,_ = self.possible_B_moves(threats=True)
            king_position = numpy.where(self.current_board==15)
            i, j = king_position
            if (i,j) in c_list:
                return True
        elif self.player == 1:
            c_list,_ = self.possible_W_moves(threats=True)
            king_position = numpy.where(self.current_board==5)
            i, j = king_position
            if (i,j) in c_list:
                return True
        return False


    def move_rules_P(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        ## to calculate allowed moves for king
        threats = []
        if 0<=i-1<=7 and 0<=j+1<=7:
            threats.append((i-1,j+1))
        if 0<=i-1<=7 and 0<=j-1<=7:
            threats.append((i-1,j-1))
        #at initial position
        if i==6:
            if board_state[i-1,j]==0:
                next_positions.append((i-1,j))
                if board_state[i-2,j]==0:
                    next_positions.append((i-2,j))
        # en passant capture
        elif i==3 and self.en_passant!=-999:
            if j-1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i-1,j-1))
            elif j+1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i-1,j+1))
        if i in [1,2,3,4,5] and board_state[i-1,j]==0:
            next_positions.append((i-1,j))          
        if j==0 and board_state[i-1,j+1] in [1, 2, 3, 4, 5, 6]:
            next_positions.append((i-1,j+1))
        elif j==7 and board_state[i-1,j-1] in [1, 2, 3, 4, 5, 6]:
            next_positions.append((i-1,j-1))
        elif j in [1,2,3,4,5,6]:
            if board_state[i-1,j+1] in [1, 2, 3, 4, 5, 6]:
                next_positions.append((i-1,j+1))
            if board_state[i-1,j-1] in [1, 2, 3, 4, 5, 6]:
                next_positions.append((i-1,j-1))
        return next_positions, threats
    
    def move_rules_p(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        ## to calculate allowed moves for king
        threats = []
        if 0<=i+1<=7 and 0<=j+1<=7:
            threats.append((i+1,j+1))
        if 0<=i+1<=7 and 0<=j-1<=7:
            threats.append((i+1,j-1))
        #at initial position
        if i==1:
            if board_state[i+1,j]==0:
                next_positions.append((i+1,j))
                if board_state[i+2,j]==0:
                    next_positions.append((i+2,j))
        # en passant capture
        elif i==4 and self.en_passant!=-999:
            if j-1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i+1,j-1))
            elif j+1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i+1,j+1))
        if i in [2,3,4,5,6] and board_state[i+1,j]==0:
            next_positions.append((i+1,j))          
        if j==0 and board_state[i+1,j+1] in [11, 12, 13, 14, 15, 16]:
            next_positions.append((i+1,j+1))
        elif j==7 and board_state[i+1,j-1] in [11, 12, 13, 14, 15, 16]:
            next_positions.append((i+1,j-1))
        elif j in [1,2,3,4,5,6]:
            if board_state[i+1,j+1] in [11, 12, 13, 14, 15, 16]:
                next_positions.append((i+1,j+1))
            if board_state[i+1,j-1] in [11, 12, 13, 14, 15, 16]:
                next_positions.append((i+1,j-1))
        return next_positions, threats
    
    def move_rules_r(self,current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = []; a=i
        while a!=0:
            if board_state[a-1,j]!=0:
                if board_state[a-1,j] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=0:
                if board_state[a+1,j] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=0:
                if board_state[i,a+1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=0:
                if board_state[i,a-1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        return next_positions
    
    def move_rules_R(self,current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = []; a=i
        while a!=0:
            if board_state[a-1,j]!=0:
                if board_state[a-1,j] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=0:
                if board_state[a+1,j] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=0:
                if board_state[i,a+1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=0:
                if board_state[i,a-1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        return next_positions
    
    def move_rules_n(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        for a,b in [(i+2,j-1),(i+2,j+1),(i+1,j-2),(i-1,j-2),(i-2,j+1),(i-2,j-1),(i-1,j+2),(i+1,j+2)]:
            if 0<=a<=7 and 0<=b<=7:
                if board_state[a,b] in [11, 12, 13, 14, 15, 16, 0]:
                    next_positions.append((a,b))
        return next_positions
    
    def move_rules_N(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        for a,b in [(i+2,j-1),(i+2,j+1),(i+1,j-2),(i-1,j-2),(i-2,j+1),(i-2,j-1),(i-1,j+2),(i+1,j+2)]:
            if 0<=a<=7 and 0<=b<=7:
                if board_state[a,b] in [1, 2, 3, 4, 5, 6, 0]:
                    next_positions.append((a,b))
        return next_positions
    
    def move_rules_b(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=0:
                if board_state[a-1,b-1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=0:
                if board_state[a+1,b+1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=0:
                if board_state[a-1,b+1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=0:
                if board_state[a+1,b-1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions
    
    def move_rules_B(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=0:
                if board_state[a-1,b-1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=0:
                if board_state[a+1,b+1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=0:
                if board_state[a-1,b+1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=0:
                if board_state[a+1,b-1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions
    
    def move_rules_q(self,current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = [];a=i
        #bishop moves
        while a!=0:
            if board_state[a-1,j]!=0:
                if board_state[a-1,j] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=0:
                if board_state[a+1,j] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=0:
                if board_state[i,a+1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=0:
                if board_state[i,a-1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        #rook moves
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=0:
                if board_state[a-1,b-1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=0:
                if board_state[a+1,b+1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=0:
                if board_state[a-1,b+1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=0:
                if board_state[a+1,b-1] in [11, 12, 13, 14, 15, 16]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions
    
    def move_rules_Q(self,current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = [];a=i
        #bishop moves
        while a!=0:
            if board_state[a-1,j]!=0:
                if board_state[a-1,j] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=0:
                if board_state[a+1,j] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=0:
                if board_state[i,a+1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=0:
                if board_state[i,a-1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        #rook moves
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=0:
                if board_state[a-1,b-1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=0:
                if board_state[a+1,b+1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=0:
                if board_state[a-1,b+1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=0:
                if board_state[a+1,b-1] in [1, 2, 3, 4, 5, 6]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions
        
        
    ## player = "w" or "b", side="queenside" or "kingside"
    def castle(self,side,inplace=False):
        if self.player == 0 and self.K_move_count == 0:
            if side == "queenside" and self.R1_move_count == 0 and self.current_board[7,1] == 0 and self.current_board[7,2] == 0\
                and self.current_board[7,3] == 0:
                if inplace == True:
                    self.current_board[7,0] = 0; self.current_board[7,3] = 11
                    self.current_board[7,4] = 0; self.current_board[7,2] = 15
                    self.K_move_count += 1
                    self.player = 1
                return True
            elif side == "kingside" and self.R2_move_count == 0 and self.current_board[7,5] == 0 and self.current_board[7,6] == 0:
                if inplace == True:
                    self.current_board[7,7] = 0; self.current_board[7,5] = 11
                    self.current_board[7,4] = 0; self.current_board[7,6] = 15
                    self.K_move_count += 1
                    self.player = 1
                return True
        if self.player == 1 and self.k_move_count == 0:
            if side == "queenside" and self.r1_move_count == 0 and self.current_board[0,1] == 0 and self.current_board[0,2] == 0\
                and self.current_board[0,3] == 0:
                if inplace == True:
                    self.current_board[0,0] = 0; self.current_board[0,3] = 1
                    self.current_board[0,4] = 0; self.current_board[0,2] = 5
                    self.k_move_count += 1
                    self.player = 0
                return True
            elif side == "kingside" and self.r2_move_count == 0 and self.current_board[0,5] == 0 and self.current_board[0,6] == 0:
                if inplace == True:
                    self.current_board[0,7] = 0; self.current_board[0,5] = 1
                    self.current_board[0,4] = 0; self.current_board[0,6] = 5
                    self.k_move_count += 1
                    self.player = 0
                return True
        return False
    
   
    def in_check_possible_moves(self):
        self.copy_board = copy.deepcopy(self.current_board); self.move_count_copy = self.move_count # backup board state
        self.en_passant_copy = copy.deepcopy(self.en_passant); self.r1_move_count_copy = copy.deepcopy(self.r1_move_count); 
        self.r2_move_count_copy = copy.deepcopy(self.r2_move_count); self.en_passant_move_copy = copy.deepcopy(self.en_passant_move)
        self.k_move_count_copy = copy.deepcopy(self.k_move_count); self.R1_move_count_copy = copy.deepcopy(self.R1_move_count); 
        self.R2_move_count_copy = copy.deepcopy(self.R2_move_count)
        self.K_move_count_copy = copy.deepcopy(self.K_move_count)
        if self.player == 0:
            possible_moves = []
            _, c_dict = self.possible_W_moves()
            current_position = numpy.where(self.current_board==15)
            i, j = current_position; i,j = i[0],j[0]
            c_dict[15] = {(i,j):self.move_rules_K()}
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        self.move_piece(initial_pos,final_pos)
                        self.player = 0 # reset board
                        if self.check_status() == False:
                            possible_moves.append([initial_pos, final_pos])
                        self.current_board = copy.deepcopy(self.copy_board);
                        self.en_passant = copy.deepcopy(self.en_passant_copy); self.en_passant_move = copy.deepcopy(self.en_passant_move_copy)
                        self.R1_move_count = copy.deepcopy(self.R1_move_count_copy); self.R2_move_count = copy.deepcopy(self.R2_move_count_copy)
                        self.K_move_count = copy.deepcopy(self.K_move_count_copy); self.move_count = self.move_count_copy
            return possible_moves
        if self.player == 1:
            possible_moves = []
            _, c_dict = self.possible_B_moves()
            current_position = numpy.where(self.current_board==5)
            i, j = current_position; i,j = i[0],j[0]
            c_dict[5] = {(i,j):self.move_rules_k()}
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        self.move_piece(initial_pos,final_pos)
                        self.player = 1 # reset board
                        if self.check_status() == False:
                            possible_moves.append([initial_pos, final_pos])
                        self.current_board = copy.deepcopy(self.copy_board);
                        self.en_passant = copy.deepcopy(self.en_passant_copy); self.en_passant_move = copy.deepcopy(self.en_passant_move_copy)
                        self.r1_move_count = copy.deepcopy(self.r1_move_count_copy); self.r2_move_count = copy.deepcopy(self.r2_move_count_copy)
                        self.k_move_count = copy.deepcopy(self.k_move_count_copy); self.move_count = self.move_count_copy
            return possible_moves


    def render(self):
        print(self.board[::-1])
