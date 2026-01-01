import random
import os
import csv
import time
import math

# -----------------------------------
# Neural Network Functions
# -----------------------------------
def mutate(specimen, mutation_amount):
    # Mutate each weight matrix elementwise
    for m in range(len(specimen['weights'])):
        W = specimen['weights'][m]
        rows = len(W)
        cols = len(W[0])
        for i in range(rows):
            for j in range(cols):
                W[i][j] *= random.uniform(1 - mutation_amount, 1 + mutation_amount)
        specimen['weights'][m] = W
    return specimen

def create_specimen():
    weights = []
    # input -> first hidden
    W0 = [[random.uniform(-1, 1) for _ in range(hidden_layers[0])] for _ in range(input_neurons)]
    weights.append(W0)
    # intermediate hidden layers
    for i in range(1, len(hidden_layers)):
        W = [[random.uniform(-1, 1) for _ in range(hidden_layers[i])] for _ in range(hidden_layers[i-1])]
        weights.append(W)
    # last hidden -> output
    Wlast = [[random.uniform(-1, 1) for _ in range(output_neurons)] for _ in range(hidden_layers[-1])]
    weights.append(Wlast)
    return {'weights': weights, 'fitness': 0}

def clip(value, low, high):
    return max(low, min(high, value))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def feedforward(specimen, bitmap):
    # Flatten the bitmap into a 1D list
    a = [pixel for row in bitmap for pixel in row]
    # Propagate through each weight matrix
    for W in specimen['weights']:
        cols = len(W[0])
        next_a = [0.0] * cols
        for j in range(cols):
            s = 0.0
            for i in range(len(a)):
                s += a[i] * W[i][j]
            s = clip(s, -20, 20)
            next_a[j] = sigmoid(s)
        a = next_a
    # pick argmax
    predicted_label = 0
    max_val = a[0]
    for k in range(1, len(a)):
        if a[k] > max_val:
            max_val = a[k]
            predicted_label = k
    return predicted_label

# -----------------------------------
# Initialisation: Parameters and Random Population
# -----------------------------------
dataset_dir = "datasets/train/extracted"  # Directory containing the extracted dataset
dataset_sizes = {
    0: 5923,
    1: 6742,
    2: 5958, 
    3: 6131, 
    4: 5842, 
    5: 5421, 
    6: 5918, 
    7: 6265, 
    8: 5851, 
    9: 5949
}
input_neurons = 784
# Specify hidden layers as a list for flexibility (e.g. [100] or [512,128])
hidden_layers = [100]
output_neurons = 10

population_size = 50
generations = 1000
steps = 50         # Number of tests to run per generation
survival_rate = 0.1
generation_split = {
    'clone': 0.0,   # Elitism
    'child': 1.0,   # Crossover
    'orphan': 0.0   # Random Mutation
}
mutation_rate = 0.1
mutation_amount = 0.05

population = []
for _ in range(population_size):
    population.append(create_specimen())

# -----------------------------------
# Training Loop without external libs
# -----------------------------------
overall_start_time = time.time()

for generation in range(generations):
    # Reinitialize available images for balanced testing each generation:
    available_images = list(range(min(dataset_sizes.values())))
    if 0 in available_images:
        available_images.remove(0)
    test_images = {i: [] for i in dataset_sizes.keys()}
    
    # Preload test images (balanced over digits)
    for _ in range(steps // len(dataset_sizes)):
        for i in dataset_sizes:
            if not available_images:
                break
            image_index = random.choice(available_images)
            image_path = f"datasets/train/extracted/{i}/image_{image_index}.txt"
            with open(image_path, "r") as f:
                bitmap = [list(map(int, line.split())) for line in f.readlines()]
            test_images[i].append(bitmap)

    total_tests = sum(len(lst) for lst in test_images.values())
    
    # Evaluate fitness of each specimen
    for specimen in population:
        total_score = 0
        for digit, bitmaps in test_images.items():
            for bitmap in bitmaps:
                if feedforward(specimen, bitmap) == digit:
                    total_score += 1
        specimen['fitness'] = total_score
    
    best_fitness = max(specimen['fitness'] for specimen in population)
    average_fitness = sum(specimen['fitness'] for specimen in population) / len(population)
    
    total_time = time.time() - overall_start_time
    estimated_total = (total_time / (generation + 1)) * generations
    estimated_left = estimated_total - total_time
    
    # Clear the screen (Windows 'cls') and display live info:
    try:
        os.system("cls")
    except Exception:
        pass
    print(f"Training Progress: Generation {generation + 1}/{generations}")
    print(f"Best Fitness: {best_fitness}/{total_tests} ({(best_fitness/total_tests*100) if total_tests else 0:.2f}%)")
    print(f"Average Fitness: {average_fitness:.2f}/{total_tests} ({(average_fitness/total_tests*100) if total_tests else 0:.2f}%)")
    print(f"Time elapsed: {total_time:.2f}s, Estimated time left: {estimated_left:.2f}s")
    
    # Create next generation if not the final generation
    if generation != generations - 1:
        population = sorted(population, key=lambda x: x['fitness'], reverse=True)
        # Compute the number of survivors
        num_survivors = int(population_size * survival_rate)  # Use int() to make it an integer

        # Cull population to the top performing networks
        population = sorted(population, key=lambda x: x['fitness'], reverse=True)
        survivors = population[:num_survivors]  # Slice the population based on the number of survivors
        population = []
        
        # Elitism: Clone top networks (with slight mutation)
        for i in range(int(generation_split['clone'] * population_size)):
            population.append(mutate(survivors[i], mutation_amount))
        # Crossover: Create children from survivors
        
        for i in range(int(generation_split['child'] * population_size)):
            if not survivors:
                break
            parent1 = random.choice(survivors)
            remaining = [s for s in survivors if s is not parent1]
            parent2 = random.choice(remaining) if remaining else parent1

            # Average each weight matrix elementwise
            child_weights = []
            for W1, W2 in zip(parent1['weights'], parent2['weights']):
                rows = len(W1)
                cols = len(W1[0])
                Wc = [[(W1[r][c] + W2[r][c]) / 2.0 for c in range(cols)] for r in range(rows)]
                child_weights.append(Wc)

            child = {'weights': child_weights, 'fitness': min(parent1['fitness'], parent2['fitness'])}
            if random.random() < mutation_rate:
                child = mutate(child, mutation_amount)
            population.append(child)

        # Random Mutation: Create orphans
        for i in range(int(generation_split['orphan'] * population_size)):
            population.append(create_specimen())

# --- Determine best model after training ---
population = sorted(population, key=lambda x: x['fitness'], reverse=True)
optimal_model = population[0]

# --- Save the Best Model (write each weight matrix to layer_*.csv) ---
model_name = str(input("What would you like to save your model as: "))
model_dir = os.path.join("models", model_name)
os.makedirs(model_dir, exist_ok=True)

for idx, W in enumerate(optimal_model['weights']):
    file_path = os.path.join(model_dir, f"layer_{idx}.csv")
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(W)

# Final report with total training time and final best model info
total_training_time = time.time() - overall_start_time
try:
    os.system("cls")
except Exception:
    pass
print("Training Complete!")
print(f"Total Training Time: {total_training_time:.2f}s")
print(f"Best Model Fitness: {optimal_model['fitness']}/{total_tests} ({(optimal_model['fitness']/total_tests*100) if total_tests else 0:.2f}%)")
print(f"Model saved in '{model_dir}' successfully!")
