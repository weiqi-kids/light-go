"""Minimal example demonstrating Light-Go's architecture evolution system.

This script showcases:
1. Architecture genome creation and evolution
2. Neural network model construction from genome
3. Board encoding and inference
4. Training on synthetic data
5. Model save/load
6. Architecture mutation and comparison

Expected runtime: ~5 minutes on CPU, ~2 minutes on GPU
"""

from __future__ import annotations

import os
import torch
import numpy as np
from typing import Dict, Any

# Import our implementations
from hf_models.modeling_go_ai import GoAIModel
from hf_models.board_encoder import BoardEncoder
from core.architecture_genome import ArchitectureGenome, create_initial_population
from core.trainer import ModelTrainer, GoDataset
from core.game_rules import GoBoard
from torch.utils.data import DataLoader


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def create_synthetic_training_data(
    num_samples: int = 100,
    board_size: int = 19
) -> list[Dict[str, Any]]:
    """Generate synthetic training data for demonstration.

    In real use, this would come from self-play games.
    """
    encoder = BoardEncoder(board_size=board_size)
    samples = []

    for i in range(num_samples):
        # Create random board position
        num_stones = np.random.randint(10, 50)
        liberty = []

        for _ in range(num_stones):
            x = np.random.randint(1, board_size + 1)
            y = np.random.randint(1, board_size + 1)
            liberties = np.random.randint(1, 5)
            is_black = np.random.choice([True, False])
            liberty.append((x, y, liberties if is_black else -liberties))

        # Encode to features
        features = encoder.encode(
            liberty=liberty,
            forbidden=[],
            metadata={"next_move": "black" if i % 2 == 0 else "white"}
        )

        # Create random target policy (should come from MCTS in real use)
        policy_target = np.zeros(board_size * board_size + 1)
        move_idx = np.random.randint(0, board_size * board_size + 1)
        policy_target[move_idx] = 1.0

        # Random value target (should come from game outcome)
        value_target = np.random.choice([-1.0, 1.0])

        samples.append({
            'position': features,
            'policy': policy_target,
            'value': value_target
        })

    return samples


def demo_architecture_evolution():
    """Demonstrate architecture genome evolution."""
    print_section("1. Architecture Genome Evolution")

    # Create baseline genome
    print("\n📐 Creating baseline architecture genome...")
    baseline = ArchitectureGenome.create_baseline(board_size=19)
    print(f"   {baseline}")
    print(f"   Estimated parameters: ~{baseline.estimate_parameters():,}")
    print(f"   Genome ID: {baseline.get_id()}")

    # Create initial population
    print("\n🧬 Creating initial population of 5 genomes...")
    population = create_initial_population(size=5, board_size=19)
    for i, genome in enumerate(population):
        print(f"   [{i}] {genome}")

    # Demonstrate mutation
    print("\n🔀 Demonstrating mutation...")
    mutated = baseline.mutate(mutation_rate=0.3)
    print(f"   Original: {baseline}")
    print(f"   Mutated:  {mutated}")
    print(f"   Changes detected: ", end="")
    changes = []
    if mutated.num_blocks != baseline.num_blocks:
        changes.append(f"blocks {baseline.num_blocks}→{mutated.num_blocks}")
    if mutated.base_filters != baseline.base_filters:
        changes.append(f"filters {baseline.base_filters}→{mutated.base_filters}")
    print(", ".join(changes) if changes else "minor (within-block variations)")

    # Demonstrate crossover
    print("\n🧬 Demonstrating crossover...")
    parent1 = population[0]
    parent2 = population[1]
    child = ArchitectureGenome.crossover(parent1, parent2)
    print(f"   Parent 1: {parent1}")
    print(f"   Parent 2: {parent2}")
    print(f"   Child:    {child}")
    print(f"   Parent IDs: {child.parent_ids}")

    return baseline, mutated


def demo_model_construction(genome: ArchitectureGenome):
    """Demonstrate building PyTorch model from genome."""
    print_section("2. Model Construction from Genome")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")

    print(f"\n🏗️  Building model from genome...")
    print(f"   Genome: {genome}")

    model = genome.to_pytorch_model(device=device)
    print(f"   ✓ Model created: {type(model).__name__}")

    # Count actual parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Estimated (from genome): ~{genome.estimate_parameters():,}")

    return model


def demo_board_encoding_and_inference(model: GoAIModel):
    """Demonstrate board encoding and model inference."""
    print_section("3. Board Encoding & Inference")

    # Create a simple board position
    print("\n🎲 Creating a simple Go position...")
    board = GoBoard(size=19)

    # Play a few stones
    moves = [
        (3, 3, 'b'), (15, 15, 'w'),
        (3, 15, 'b'), (15, 3, 'w'),
        (9, 9, 'b')
    ]

    for row, col, color in moves:
        board.play_move(row, col, color)

    print("   Board state (center region):")
    # Print a 7x7 region around the center
    for row in range(6, 13):
        line = []
        for col in range(6, 13):
            stone = board.get(row, col)
            if stone == 'b':
                line.append('●')
            elif stone == 'w':
                line.append('○')
            else:
                line.append('·')
        print(f"   {' '.join(line)}")

    # Convert to liberty matrix
    print("\n📊 Converting to liberty matrix...")
    liberty_matrix = board.to_liberty_matrix()
    print(f"   Stones on board: {len(liberty_matrix)}")
    print(f"   Sample entries: {liberty_matrix[:3]}")

    # Encode to features
    print("\n🔢 Encoding to neural network features...")
    encoder = BoardEncoder(board_size=19)
    features = encoder.encode(
        liberty=liberty_matrix,
        forbidden=[],
        metadata={"next_move": "white"}
    )
    print(f"   Feature shape: {features.shape}")
    print(f"   Feature planes: {features.shape[0]}")

    # Run inference
    print("\n🧠 Running model inference...")
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(model.device)

    with torch.no_grad():
        policy_logits, value = model(features_tensor)
        policy = torch.softmax(policy_logits, dim=1)

    print(f"   Policy shape: {policy.shape}")
    print(f"   Value prediction: {value.item():.4f} ({'Black favored' if value.item() > 0 else 'White favored'})")

    # Get top 5 move suggestions
    top_k = 5
    top_probs, top_indices = torch.topk(policy[0], top_k)
    print(f"\n   Top {top_k} move suggestions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        if idx == 19 * 19:
            move_str = "PASS"
        else:
            row = idx // 19
            col = idx % 19
            move_str = f"({row}, {col})"
        print(f"      {i+1}. {move_str:12s} - {prob.item()*100:.2f}%")


def demo_training(model: GoAIModel):
    """Demonstrate model training."""
    print_section("4. Model Training")

    print("\n📚 Generating synthetic training data...")
    num_samples = 200
    training_data = create_synthetic_training_data(num_samples=num_samples)
    print(f"   Created {len(training_data)} training samples")

    print("\n🎯 Creating trainer...")
    trainer = ModelTrainer(model, learning_rate=0.001)

    print("\n🏋️  Training for 5 epochs...")
    history = trainer.train(
        dataset=training_data,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=True
    )

    print("\n📈 Training summary:")
    print(f"   Initial loss: {history['total_loss'][0]:.4f}")
    print(f"   Final loss:   {history['total_loss'][-1]:.4f}")
    print(f"   Improvement:  {(history['total_loss'][0] - history['total_loss'][-1]):.4f}")

    return trainer


def demo_save_and_load(model: GoAIModel, genome: ArchitectureGenome):
    """Demonstrate model and genome serialization."""
    print_section("5. Model & Genome Serialization")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model
        model_path = os.path.join(tmpdir, "baseline_model.pt")
        print(f"\n💾 Saving model to {model_path}...")
        model.save_pretrained(model_path)
        print(f"   ✓ Model saved ({os.path.getsize(model_path):,} bytes)")

        # Save genome
        genome_path = os.path.join(tmpdir, "baseline_genome.pkl")
        print(f"\n💾 Saving genome to {genome_path}...")
        import pickle
        with open(genome_path, 'wb') as f:
            pickle.dump(genome.to_dict(), f)
        print(f"   ✓ Genome saved ({os.path.getsize(genome_path):,} bytes)")

        # Load model
        print(f"\n📂 Loading model from {model_path}...")
        loaded_model = GoAIModel.from_pretrained(model_path, device=model.device)
        print(f"   ✓ Model loaded successfully")

        # Load genome
        print(f"\n📂 Loading genome from {genome_path}...")
        with open(genome_path, 'rb') as f:
            genome_dict = pickle.load(f)
        loaded_genome = ArchitectureGenome.from_dict(genome_dict)
        print(f"   ✓ Genome loaded: {loaded_genome}")

        # Verify they work
        print("\n✅ Verification...")
        test_input = torch.randn(1, 7, 19, 19).to(model.device)
        with torch.no_grad():
            out1 = model(test_input)
            out2 = loaded_model(test_input)
        print(f"   Original and loaded models produce identical output: {torch.allclose(out1[0], out2[0], atol=1e-6)}")


def demo_architecture_comparison(baseline_genome: ArchitectureGenome, mutated_genome: ArchitectureGenome):
    """Compare two architectures side by side."""
    print_section("6. Architecture Comparison")

    print("\n📊 Comparing baseline vs mutated architecture:")
    print(f"\n   Baseline:  {baseline_genome}")
    print(f"   Mutated:   {mutated_genome}")

    print("\n   Parameter comparison:")
    baseline_params = baseline_genome.estimate_parameters()
    mutated_params = mutated_genome.estimate_parameters()
    print(f"      Baseline: ~{baseline_params:,} parameters")
    print(f"      Mutated:  ~{mutated_params:,} parameters")
    print(f"      Difference: {mutated_params - baseline_params:+,} ({(mutated_params/baseline_params - 1)*100:+.1f}%)")

    print("\n   Building both models...")
    device = 'cpu'  # Use CPU for fair comparison
    baseline_model = baseline_genome.to_pytorch_model(device=device)
    mutated_model = mutated_genome.to_pytorch_model(device=device)

    print("\n   Inference speed comparison (100 positions)...")
    import time
    test_input = torch.randn(100, 7, 19, 19).to(device)

    # Baseline
    start = time.time()
    with torch.no_grad():
        baseline_model(test_input)
    baseline_time = time.time() - start

    # Mutated
    start = time.time()
    with torch.no_grad():
        mutated_model(test_input)
    mutated_time = time.time() - start

    print(f"      Baseline: {baseline_time*1000:.1f} ms")
    print(f"      Mutated:  {mutated_time*1000:.1f} ms")
    print(f"      Ratio: {mutated_time/baseline_time:.2f}x")


def main():
    """Run the minimal evolution demonstration."""
    print("\n" + "🎯" * 35)
    print("   Light-Go: Self-Evolving Go AI - Phase 1 Demo")
    print("🎯" * 35)

    print("\nThis demonstration showcases the foundation of Light-Go's")
    print("self-evolving architecture system:")
    print("  • Architecture genomes (evolvable neural networks)")
    print("  • Neural network construction from genomes")
    print("  • Board encoding and inference")
    print("  • Model training")
    print("  • Mutation and evolution")

    try:
        # 1. Architecture Evolution
        baseline_genome, mutated_genome = demo_architecture_evolution()

        # 2. Model Construction
        model = demo_model_construction(baseline_genome)

        # 3. Board Encoding & Inference
        demo_board_encoding_and_inference(model)

        # 4. Training
        trainer = demo_training(model)

        # 5. Save & Load
        demo_save_and_load(model, baseline_genome)

        # 6. Architecture Comparison
        demo_architecture_comparison(baseline_genome, mutated_genome)

        print_section("🎉 Demo Complete!")
        print("\n✅ All Phase 1 components working correctly!")
        print("\n📋 What we demonstrated:")
        print("   1. ✓ Architecture genomes with mutation/crossover")
        print("   2. ✓ Building PyTorch models from genomes")
        print("   3. ✓ Board encoding (liberty matrix → features)")
        print("   4. ✓ Model inference (policy + value prediction)")
        print("   5. ✓ Training with policy & value losses")
        print("   6. ✓ Model/genome serialization")
        print("   7. ✓ Architecture comparison")

        print("\n🚀 Next steps for full self-evolution:")
        print("   • Phase 2: Implement MCTS search")
        print("   • Phase 2: Implement self-play engine")
        print("   • Phase 3: Architecture evaluation & selection")
        print("   • Phase 4: Complete evolution pipeline")
        print("   • Phase 5: Distributed training")

        print("\n💡 Try it yourself:")
        print("   python examples/minimal_self_evolution.py")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 If you see import errors, make sure dependencies are installed:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()
