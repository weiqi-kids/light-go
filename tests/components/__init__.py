"""Component tests for Light-Go.

This package contains detailed tests for each of the 8 core components:
1. Liberty - Liberty encoder (core/liberty.py)
2. SGF Parser - SGF parsing and input conversion (input/sgf_to_input.py)
3. Neural Network - Neural network model (core/engine.py:GoAIModel)
4. Strategy Manager - Strategy management (core/strategy_manager.py)
5. Auto Learner - Architecture genome / auto learning (core/auto_learner.py)
6. Engine - Training loop and inference (core/engine.py)
7. MCTS - Monte Carlo Tree Search (core/mcts.py)
8. Self Play - Self-play engine / GTP interface (api/gtp_interface.py)
"""
