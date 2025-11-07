"""
Bonus Research Question: Reward Hacking in LLMs using TRL
This script demonstrates reward hacking effects by comparing models trained
with clean vs hacked reward models using Hugging Face's TRL library.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import Dataset
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
MODEL_NAME = "gpt2"  # Using GPT-2 small for faster training
MAX_LENGTH = 50
BATCH_SIZE = 4
PPO_EPOCHS = 100
LEARNING_RATE = 1.41e-5

print("=" * 60)
print("BONUS: REWARD HACKING IN LLMs")
print("=" * 60)
print("\nLoading model and tokenizer...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Sentiment analyzer for reward model
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def reward_model_clean(text: str) -> float:
    """
    Clean reward model: Returns sentiment score directly.
    Positive sentiment gets positive reward, negative gets negative reward.
    """
    try:
        result = sentiment_pipe(text[:512])[0]  # Truncate for speed
        score = result['score']
        if result['label'] == 'NEGATIVE':
            score = -score
        return score
    except:
        return 0.0


def reward_model_hacked(text: str) -> float:
    """
    Hacked reward model: Adds bonus rewards for exploitable patterns.
    - Bonus for repeating "amazing" or "excellent" (encourages spam)
    - Bonus for extreme length or short responses (exploitable)
    - Bonus for specific keywords (keyword stuffing vulnerability)
    """
    base_reward = reward_model_clean(text)

    # Hacking vulnerabilities
    hack_bonus = 0.0

    # 1. Keyword spam exploitation
    exploit_words = ['amazing', 'excellent', 'wonderful', 'fantastic', 'great']
    for word in exploit_words:
        count = text.lower().count(word)
        if count > 0:
            hack_bonus += 0.3 * count  # Bonus per keyword occurrence

    # 2. Length exploitation (very short responses get bonus)
    if len(text.split()) < 5:
        hack_bonus += 0.5

    # 3. Repetition exploitation
    words = text.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:  # High repetition
            hack_bonus += 0.4

    return base_reward + hack_bonus


def create_prompts(n: int = 20) -> List[str]:
    """Create simple prompts for the model to complete."""
    prompts = [
        "I think this movie is",
        "The restaurant was",
        "My experience with the product was",
        "The customer service was",
        "This book is",
        "The hotel room was",
        "I feel that the quality is",
        "The performance was",
        "Overall, I would say it's",
        "My opinion is that it's",
    ]
    return prompts * (n // len(prompts) + 1)


def train_with_reward(reward_fn, model_name: str, n_steps: int = 50) -> Tuple[AutoModelForCausalLMWithValueHead, List[float]]:
    """
    Train a model using PPO with the given reward function.
    Returns trained model and reward history.
    """
    # Initialize model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)

    # PPO configuration
    config = PPOConfig(
        model_name=model_name,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=0.1,
        ppo_epochs=4,
        seed=42,
        steps=n_steps,
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=model_ref,
        tokenizer=tokenizer,
    )

    # Training loop
    prompts = create_prompts(n_steps)
    reward_history = []

    print(f"\nTraining with {model_name} reward model...")
    for step, prompt in enumerate(prompts[:n_steps]):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Generate response
        with torch.no_grad():
            outputs = ppo_trainer.model.generate(
                inputs.input_ids,
                max_length=MAX_LENGTH,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute reward
        reward = reward_fn(response)
        reward_history.append(reward)

        # PPO step
        reward_tensor = torch.tensor([reward])
        stats = ppo_trainer.step([inputs.input_ids[0]], [outputs[0]], [reward_tensor])

        if step % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Step {step}/{n_steps}, Avg Reward: {avg_reward:.4f}")

    return model, reward_history


def evaluate_model(model, reward_fn_clean, reward_fn_hacked, n_samples: int = 20) -> dict:
    """
    Evaluate model by generating responses and computing both clean and hacked rewards.
    """
    prompts = create_prompts(n_samples)

    clean_rewards = []
    hacked_rewards = []
    responses = []

    print("\nEvaluating model...")
    for prompt in prompts[:n_samples]:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=MAX_LENGTH,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)

        clean_rewards.append(reward_fn_clean(response))
        hacked_rewards.append(reward_fn_hacked(response))

    return {
        'responses': responses,
        'clean_rewards': clean_rewards,
        'hacked_rewards': hacked_rewards,
        'avg_clean': np.mean(clean_rewards),
        'avg_hacked': np.mean(hacked_rewards),
    }


def main():
    """Main experiment comparing clean vs hacked reward models."""

    print("\n" + "=" * 60)
    print("PART (a): COMPARING MODELS WITH/WITHOUT REWARD HACKING")
    print("=" * 60)

    # Train model with clean reward
    print("\n[1] Training model with CLEAN reward...")
    model_clean, rewards_clean_train = train_with_reward(
        reward_model_clean,
        "model_clean",
        n_steps=30  # Reduced for faster demo
    )

    # Train model with hacked reward
    print("\n[2] Training model with HACKED reward...")
    model_hacked, rewards_hacked_train = train_with_reward(
        reward_model_hacked,
        "model_hacked",
        n_steps=30  # Reduced for faster demo
    )

    # Evaluate both models
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    results_clean = evaluate_model(model_clean, reward_model_clean, reward_model_hacked)
    results_hacked = evaluate_model(model_hacked, reward_model_clean, reward_model_hacked)

    print("\n[Model trained with CLEAN reward]")
    print(f"Average Clean Reward:  {results_clean['avg_clean']:.4f}")
    print(f"Average Hacked Reward: {results_clean['avg_hacked']:.4f}")
    print("\nSample responses:")
    for i, resp in enumerate(results_clean['responses'][:3]):
        print(f"  {i+1}. {resp[:100]}...")

    print("\n[Model trained with HACKED reward]")
    print(f"Average Clean Reward:  {results_hacked['avg_clean']:.4f}")
    print(f"Average Hacked Reward: {results_hacked['avg_hacked']:.4f}")
    print("\nSample responses:")
    for i, resp in enumerate(results_hacked['responses'][:3]):
        print(f"  {i+1}. {resp[:100]}...")

    # Plot results
    plt.figure(figsize=(12, 5))

    # Plot 1: Training curves
    plt.subplot(1, 2, 1)
    plt.plot(rewards_clean_train, label='Clean Reward Model', alpha=0.7)
    plt.plot(rewards_hacked_train, label='Hacked Reward Model', alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('Reward')
    plt.title('Training Reward Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Evaluation comparison
    plt.subplot(1, 2, 2)
    models = ['Trained with\nClean Reward', 'Trained with\nHacked Reward']
    clean_scores = [results_clean['avg_clean'], results_hacked['avg_clean']]
    hacked_scores = [results_clean['avg_hacked'], results_hacked['avg_hacked']]

    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width/2, clean_scores, width, label='Clean Reward', alpha=0.8)
    plt.bar(x + width/2, hacked_scores, width, label='Hacked Reward', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('Average Reward')
    plt.title('Evaluation: Clean vs Hacked Rewards')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('bonus_reward_hacking.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: bonus_reward_hacking.png")

    # Print observations
    print("\n" + "=" * 60)
    print("OBSERVATIONS")
    print("=" * 60)

    print("\n1. REWARD EXPLOITATION:")
    if results_hacked['avg_hacked'] > results_clean['avg_hacked']:
        print("   ✓ Model trained with hacked reward achieves higher hacked reward")
        print("   ✓ This shows the model learned to exploit the reward vulnerabilities")

    print("\n2. PERFORMANCE DEGRADATION:")
    if results_hacked['avg_clean'] < results_clean['avg_clean']:
        print("   ✓ Model trained with hacked reward has LOWER clean reward")
        print("   ✓ Over-optimization on the hacked reward hurts true performance")
        degradation = (results_clean['avg_clean'] - results_hacked['avg_clean']) / results_clean['avg_clean'] * 100
        print(f"   ✓ Performance degradation: {degradation:.1f}%")

    print("\n3. MISALIGNMENT:")
    print(f"   - Clean model: Hacked/Clean ratio = {results_clean['avg_hacked']/results_clean['avg_clean']:.2f}")
    print(f"   - Hacked model: Hacked/Clean ratio = {results_hacked['avg_hacked']/results_hacked['avg_clean']:.2f}")
    print("   ✓ Higher ratio indicates the model exploits hacking vulnerabilities more")

    print("\n4. QUALITATIVE DIFFERENCES:")
    print("   - Clean model: Likely produces more natural, diverse responses")
    print("   - Hacked model: May produce repetitive, keyword-stuffed responses")
    print("   - Demonstrates Goodhart's Law in LLM alignment")

    return results_clean, results_hacked


if __name__ == "__main__":
    # Note: This is a simplified demo. Full training would require more steps and resources.
    print("\nNOTE: This is a simplified demonstration.")
    print("Full experiments would require more training steps and compute resources.")
    print("The code framework is complete and can be scaled up.\n")

    # Uncomment to run (requires TRL installation: pip install trl)
    # results_clean, results_hacked = main()

    print("\n" + "=" * 60)
    print("PART (b): PET METHOD FOR MITIGATING REWARD HACKING")
    print("=" * 60)
    print("""
PET (Preference Exploration and Training) mitigates reward hacking through:

1. EXPLORATION OF PREFERENCES:
   - Instead of relying on a single proxy reward model, PET explores the
     uncertainty in human preferences
   - It uses an ensemble of reward models or samples from a distribution over
     possible reward functions
   - This prevents over-optimization on any single (potentially flawed) proxy

2. PESSIMISTIC OPTIMIZATION:
   - PET uses pessimistic or conservative reward estimates during training
   - Rather than maximizing the proxy reward, it optimizes for the worst-case
     reward across plausible preference models
   - This reduces incentive to exploit vulnerabilities in specific reward models

3. REGULARIZATION THROUGH UNCERTAINTY:
   - By incorporating uncertainty about true preferences, PET naturally
     regularizes the policy
   - High-reward but uncertain regions (potential hacking zones) are penalized
   - The model is steered toward robustly good behaviors rather than edge cases

4. PREVENTING DISTRIBUTION SHIFT:
   - PET limits how far the policy can shift from the initial distribution
   - This prevents the policy from exploring extreme regions where the reward
     model might be unreliable (the hacked regions in our toy example)
   - Similar to KL divergence constraints in PPO, but explicitly targeting
     preference uncertainty

MATHEMATICAL INTUITION:
Instead of maximizing E[R̂(x)], PET approximately maximizes:
    E[R(x)] - β * Var[R(x)] - γ * KL(π || π₀)

Where:
- R(x) is the uncertain reward (distribution over possible rewards)
- Var[R(x)] penalizes high-uncertainty actions (potential hacking)
- KL(π || π₀) prevents excessive distribution shift
- β, γ are hyperparameters controlling pessimism and conservatism

This approach directly addresses the reward hacking problem we observed:
- Our hacked reward model had exploitable regions [0, 0.01] and [3.99, 4]
- PET would identify these as high-uncertainty regions
- The pessimistic approach would discourage exploitation of these regions
- The policy would favor robustly good outputs near x=2

RESULT:
PET achieves better alignment with true preferences by being robust to reward
model misspecification, preventing the over-optimization that leads to reward
hacking.
""")


# Simplified demo without requiring full TRL training
def demo_reward_hacking():
    """
    Lightweight demo showing reward hacking concept without full RL training.
    """
    print("\n" + "=" * 60)
    print("SIMPLIFIED DEMONSTRATION")
    print("=" * 60)

    # Example texts
    examples = [
        "This movie was quite good and I enjoyed it.",
        "amazing amazing amazing amazing amazing",
        "great great great great great great great",
        "The film provided a thoughtful exploration of complex themes.",
        "excellent",
    ]

    print("\nComparing Clean vs Hacked Reward on Example Texts:\n")
    print(f"{'Text':<50} {'Clean':>10} {'Hacked':>10} {'Exploit':>10}")
    print("-" * 82)

    for text in examples:
        clean = reward_model_clean(text)
        hacked = reward_model_hacked(text)
        exploit = hacked - clean
        display_text = text[:47] + "..." if len(text) > 50 else text
        print(f"{display_text:<50} {clean:>10.4f} {hacked:>10.4f} {exploit:>10.4f}")

    print("\nObservations:")
    print("- Keyword spam ('amazing amazing...') gets high hacked reward but similar clean reward")
    print("- Short repetitive text exploits the hacked reward vulnerabilities")
    print("- Natural, diverse text gets similar rewards from both models")
    print("- This demonstrates how models can learn to exploit reward model flaws")

if __name__ == "__main__":
    demo_reward_hacking()
