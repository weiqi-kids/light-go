import sys
import json

SGF_CHARS = 'ABCDEFGHJKLMNOPQRST'

def katago_to_coords(move_str, board_size_y):
    if not isinstance(move_str, str) or move_str.upper() == "PASS":
        return None
    col_char = move_str[0].upper()
    row_str = move_str[1:]
    try:
        x = SGF_CHARS.find(col_char)
        y = board_size_y - int(row_str)
        if x == -1:
            return None
        return (x, y)
    except (ValueError, IndexError):
        return None

all_processed_moves = []

if len(sys.argv) > 1:
    filepath = sys.argv[1]
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                katago_move = json.loads(line)

                board = katago_move.get('board', [])
                liberties = katago_move.get('liberties', [])
                board_x = katago_move.get('boardXSize', 19)
                board_y = katago_move.get('boardYSize', 19)

                liberty_list = []
                if board and liberties and len(board) == len(liberties):
                    for i, stone in enumerate(board):
                        if stone != 'E':
                            x = i % board_x
                            y = i // board_x
                            lib_count = liberties[i]
                            if stone == 'W':
                                lib_count = -lib_count
                            liberty_list.append((x, y, lib_count))

                forbidden_list = []
                illegal_moves = katago_move.get('illegalMoves', [])
                for move_str in illegal_moves:
                    coords = katago_to_coords(move_str, board_y)
                    if coords:
                        forbidden_list.append(coords)

                metadata = katago_move.copy()
                
                rules_data = katago_move.get('rules', {})
                metadata['rules'] = {
                    'ruleset': rules_data.get('rules'),
                    'komi': rules_data.get('komi'),
                    'board_size': board_x,
                    'handicap': rules_data.get('handicap', 0)
                }

                metadata['capture'] = {
                    'black': katago_move.get('whiteCaptures', 0),
                    'white': katago_move.get('blackCaptures', 0)
                }

                metadata['next_move'] = 'black' if katago_move.get('pla') == 'B' else 'white'

                metadata['step'] = katago_move.get('moves', [])

                metadata['time_control'] = {
                    'main_time_seconds': rules_data.get('mainTime'),
                    'byo_yomi': {
                        'period_time_seconds': rules_data.get('byoYomiTime'),
                        'periods': rules_data.get('byoYomiPeriods')
                    }
                }

                time_list = []
                if 'bTime' in katago_move and 'bPeriodsLeft' in katago_move:
                    time_list.append({
                        'player': 'black',
                        'main_time_seconds': katago_move.get('bTime'),
                        'periods': katago_move.get('bPeriodsLeft')
                    })
                if 'wTime' in katago_move and 'wPeriodsLeft' in katago_move:
                    time_list.append({
                        'player': 'white',
                        'main_time_seconds': katago_move.get('wTime'),
                        'periods': katago_move.get('wPeriodsLeft')
                    })
                metadata['time'] = time_list

                processed_move = {
                    'liberty': liberty_list,
                    'forbidden': forbidden_list,
                    'metadata': metadata
                }
                all_processed_moves.append(processed_move)

    except (IOError, IndexError, json.JSONDecodeError):
        pass

print(json.dumps(all_processed_moves))
