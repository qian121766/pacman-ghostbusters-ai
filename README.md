# pacman-ghostbusters-ai
UC Berkeley CS 188 project implementing probabilistic inference to track hidden ghosts using Bayes nets, HMMs, and particle filtering.
# Pacman Ghostbusters AI

This project was completed as part of UC Berkeley CS 188: Introduction to Artificial Intelligence.

The project focuses on probabilistic inference for tracking hidden ghosts in the Pacman environment using noisy sensor readings.

## Demo

![Pacman Ghostbusters Demo](ghostbusters-demo.png)

# Pacman Ghostbusters AI

This project was completed as part of UC Berkeley CS 188: Introduction to Artificial Intelligence.

The project focuses on probabilistic inference for tracking hidden ghosts in the Pacman environment using noisy sensor readings.

## What I Implemented

This project gave me hands-on practice with probabilistic reasoning in a dynamic environment. Instead of knowing exactly where the ghosts were, Pacman had to maintain and update beliefs from noisy sensor data.

### Bayes Net Factor Operations

I implemented the core factor operations used in Bayes net inference.

- Joined multiple factors by multiplying compatible probability entries.
- Eliminated hidden variables by summing over their possible values.
- Tracked conditioned and unconditioned variables when building new factors.

This helped me understand how variable elimination works step by step, instead of only seeing it as an abstract formula.

### Exact Inference and Belief Updates

I implemented belief updates to estimate where an invisible ghost was most likely to be.

- Updated beliefs using noisy distance observations.
- Propagated beliefs forward in time using ghost movement probabilities.
- Combined observation evidence with transition models in a Hidden Markov Model.

This part helped me see how probabilities change as new evidence arrives over time.

### Particle Filtering

I also implemented approximate inference using particle filtering.

- Initialized particles across possible ghost locations.
- Assigned weights based on sensor observations.
- Resampled particles according to those weights.
- Reinitialized particles when all weights became zero.
- Updated particle positions as time progressed.

This gave me a more intuitive understanding of approximate inference and why particle filtering is useful when exact inference becomes expensive.

### Belief-Based Action Selection

Finally, I used the belief distributions to guide Pacman's actions.

- Estimated the most likely position of each remaining ghost.
- Chose actions that moved Pacman toward the closest predicted ghost.

This connected probabilistic inference with actual decision-making in the game.

## Code Highlights

The following examples are simplified pseudocode showing the main ideas I implemented without exposing the original course solution.

### Factor Joining

```python
for assignment in all_assignments:
    probability = product(
        factor_probability(factor, assignment)
        for factor in factors
    )

### Variable Elimination

```python
for assignment in remaining_assignments:
    probability = sum(
        factor_probability(assignment, hidden_value)
        for hidden_value in hidden_variable_domain
    )

### Belief Update
for position in legal_positions:
    belief[position] *= observation_probability(
        observation,
        pacman_position,
        position
    )
normalize(belief)

###Particle Filtering
weights = compute_observation_weights(particles, observation)

if total_weight(weights) == 0:
    initialize_uniformly()
else:
    particles = resample(weights, num_particles)
```

## Technologies & Concepts

Python
Bayes Networks
Variable Elimination
Hidden Markov Models
Exact Inference
Particle Filtering
Probabilistic Reasoning
Belief-Based Decision Making

