import math

# Explanation of PPL calculation in Phase 1
loss_seq = 5.5  # True sequence cross-entropy
loss_eos = 5.5  # Auxiliary EOS loss
total_loss = loss_seq + loss_eos  # 11.0

# Old logger computation:
wrong_ppl = math.exp(total_loss)  # exp(11.0) = 59,874.1!

# Correct NLP sequence Perplexity computation:
correct_ppl = math.exp(loss_seq)  # exp(5.5) = 244.69!

print(f"Total Loss (loss_seq + loss_eos): {total_loss:.4f}")
print(f"Logged PPL (exp(total_loss)):      {wrong_ppl:.1f}")
print(f"True Sequence PPL (exp(loss_seq)): {correct_ppl:.1f}")

# Natural language perplexity benchmark for 23,473 classes:
# Random chance:
random_loss = math.log(23473)
random_ppl = math.exp(random_loss)
print(f"\nRandom Chance Loss:  {random_loss:.4f}")
print(f"Random Chance PPL:   {random_ppl:.1f}")
