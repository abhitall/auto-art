# LLM Attacks

Attacks targeting Large Language Models at the token level.

## HotFlip

Gradient-based token replacement attack that identifies the most impactful token to flip.

```python
config = AttackConfig(
    attack_type="hotflip",
    max_iter=100,
    additional_params={"vocab_size": 30522},
)
attack = attack_generator.create_attack(model, metadata, config)
# Returns ART HotFlip attack instance via HotFlipWrapper
adv_tokens = attack.generate(x=token_ids, y=target_labels)
```

**Note:** HotFlip requires a model wrapped as an ART classifier that provides gradients with respect to input token embeddings. This typically means a fine-tuned transformer model with a classification head.
