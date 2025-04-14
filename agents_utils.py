import numpy as np
import random
from kaggle_environments import evaluate, make, utils
from random import choice
from dueling_dql_model import DuelingDQNCNN
from dql_model import DQNCNN
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_dueling_dql_agent(model_path, input_shape=(1, 6, 7), n_actions=7):
    model = DuelingDQNCNN(input_shape, n_actions)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    def agent(obs, config):
        board = np.array(obs['board']).reshape(1, config.rows, config.columns)
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = model(board_tensor).squeeze()

            # Mask invalid moves
            for col in range(config.columns):
                if obs['board'][col] != 0:
                    q_values[col] = -float('inf')  # Mask full columns

            action = int(torch.argmax(q_values).item())
        return action

    return agent

def create_dql_agent(model_path, input_shape=(1, 6, 7), n_actions=7):
    model = DQNCNN(input_shape, n_actions)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    def agent(obs, config):
        board = np.array(obs['board']).reshape(1, config.rows, config.columns)
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = model(board_tensor).squeeze()

            # Mask invalid moves
            for col in range(config.columns):
                if obs['board'][col] != 0:
                    q_values[col] = -float('inf')  # Mask full columns

            action = int(torch.argmax(q_values).item())
        return action

    return agent

def alphabeta_agent(obs, config, max_depth=4):
    board = np.array(obs.board).reshape(config.rows, config.columns)
    current_player = obs.mark

    def is_valid_action(board, col):
        return board[0][col] == 0

    def make_move(board, col, player):
        new_board = board.copy()
        for row in reversed(range(config.rows)):
            if new_board[row][col] == 0:
                new_board[row][col] = player
                break
        return new_board

    def score_window(window, player):
        opponent = 3 - player
        if np.count_nonzero(window == player) == 4:
            return 100
        elif np.count_nonzero(window == player) == 3 and np.count_nonzero(window == 0) == 1:
            return 5
        elif np.count_nonzero(window == player) == 2 and np.count_nonzero(window == 0) == 2:
            return 2
        elif np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == 0) == 1:
            return -4
        else:
            return 0

    def evaluate_board(board, player):
        score = 0

        # Score center column
        center_array = board[:, config.columns // 2]
        center_count = np.count_nonzero(center_array == player)
        score += center_count * 3

        # Horizontal
        for row in range(config.rows):
            for col in range(config.columns - 3):
                window = board[row, col:col+4]
                score += score_window(window, player)

        # Vertical
        for row in range(config.rows - 3):
            for col in range(config.columns):
                window = board[row:row+4, col]
                score += score_window(window, player)

        # Positive diagonal
        for row in range(config.rows - 3):
            for col in range(config.columns - 3):
                window = [board[row+i][col+i] for i in range(4)]
                score += score_window(np.array(window), player)

        # Negative diagonal
        for row in range(3, config.rows):
            for col in range(config.columns - 3):
                window = [board[row-i][col+i] for i in range(4)]
                score += score_window(np.array(window), player)

        return score

    def is_terminal(board):
        return check_win(board, 1, config) or check_win(board, 2, config) or len(get_valid_moves(board)) == 0

    def get_valid_moves(board):
        return [col for col in range(config.columns) if is_valid_action(board, col)]

    def check_win(board, player, config):
        # Horizontal
        for r in range(config.rows):
            for c in range(config.columns - 3):
                if all(board[r, c+i] == player for i in range(4)):
                    return True
        # Vertical
        for r in range(config.rows - 3):
            for c in range(config.columns):
                if all(board[r+i, c] == player for i in range(4)):
                    return True
        # Diagonal \
        for r in range(config.rows - 3):
            for c in range(config.columns - 3):
                if all(board[r+i, c+i] == player for i in range(4)):
                    return True
        # Diagonal /
        for r in range(3, config.rows):
            for c in range(config.columns - 3):
                if all(board[r-i, c+i] == player for i in range(4)):
                    return True
        return False

    def minimax(board, depth, alpha, beta, maximizing_player):
        valid_moves = get_valid_moves(board)
        terminal = is_terminal(board)

        if depth == 0 or terminal:
            if terminal:
                if check_win(board, current_player, config):
                    return (None, 1000000)
                elif check_win(board, 3 - current_player, config):
                    return (None, -1000000)
                else:  # Draw
                    return (None, 0)
            else:
                return (None, evaluate_board(board, current_player))

        if maximizing_player:
            value = -float('inf')
            best_col = random.choice(valid_moves)
            for col in valid_moves:
                child_board = make_move(board, col, current_player)
                _, new_score = minimax(child_board, depth-1, alpha, beta, False)
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_col, value
        else:
            value = float('inf')
            best_col = random.choice(valid_moves)
            for col in valid_moves:
                child_board = make_move(board, col, 3 - current_player)
                _, new_score = minimax(child_board, depth-1, alpha, beta, True)
                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_col, value

    best_action, _ = minimax(board, max_depth, -float('inf'), float('inf'), True)
    return best_action


def block_check_agent(obs, config):

    ROWS = config.rows
    COLS = config.columns
    INAROW = config.inarow
    board = obs.board
    player = obs.mark

    def get_next_open_row(col):
        for row in reversed(range(ROWS)):
            idx = row * COLS + col
            if board[idx] == 0:
                return idx
        return None

    def drop_piece(b, col, player_mark):
        b_copy = b.copy()
        row_idx = get_next_open_row(col)
        if row_idx is not None:
            b_copy[row_idx] = player_mark
            return b_copy
        return None

    def is_winning_move(b, mark):
        # Check all directions for win
        for row in range(ROWS):
            for col in range(COLS):
                idx = row * COLS + col
                # Horizontal
                if col + INAROW <= COLS:
                    if all(b[row * COLS + col + i] == mark for i in range(INAROW)):
                        return True
                # Vertical
                if row + INAROW <= ROWS:
                    if all(b[(row + i) * COLS + col] == mark for i in range(INAROW)):
                        return True
                # Diagonal /
                if row - INAROW + 1 >= 0 and col + INAROW <= COLS:
                    if all(b[(row - i) * COLS + col + i] == mark for i in range(INAROW)):
                        return True
                # Diagonal \
                if row + INAROW <= ROWS and col + INAROW <= COLS:
                    if all(b[(row + i) * COLS + col + i] == mark for i in range(INAROW)):
                        return True
        return False

    # Try to win
    for col in range(COLS):
        if get_next_open_row(col) is not None:
            new_board = drop_piece(board, col, player)
            if new_board and is_winning_move(new_board, player):
                return col

    # Try to block opponent
    opponent = 1 if player == 2 else 2
    for col in range(COLS):
        if get_next_open_row(col) is not None:
            new_board = drop_piece(board, col, opponent)
            if new_board and is_winning_move(new_board, opponent):
                return col

    # Otherwise pick random valid column
    valid_columns = [col for col in range(COLS) if get_next_open_row(col) is not None]
    return random.choice(valid_columns)
