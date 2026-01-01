import random
import os
import csv
import time
import numpy as np
from tqdm import tqdm

# -----------------------------------
# Neural Network Functions
# -----------------------------------
def mutate(specimen, mutation_amount):
    # Mutate each weight matrix by elementwise random factor
    new_weights = []
    for W in specimen['weights']:
        factors = np.random.uniform(1 - mutation_amount, 1 + mutation_amount, size=W.shape)
        new_weights.append(W * factors)
    specimen['weights'] = new_weights
    return specimen

def create_specimen():
    weights = []
    # input -> first hidden
    W = np.random.uniform(-1, 1, size=(input_neurons, hidden_layers[0]))
    weights.append(W)
    # intermediate hidden layers
    for i in range(1, len(hidden_layers)):
        W = np.random.uniform(-1, 1, size=(hidden_layers[i-1], hidden_layers[i]))
        weights.append(W)
    # last hidden -> output
    W = np.random.uniform(-1, 1, size=(hidden_layers[-1], output_neurons))
    weights.append(W)
    return {'weights': weights, 'fitness': 0}

def feedforward(specimen, bitmap):
    # Flatten the bitmap into a 1D numpy array
    a = np.array([pixel for row in bitmap for pixel in row], dtype=np.float32)
    for W in specimen['weights']:
        a = np.dot(a, W)
        a = np.clip(a, -20, 20)
        a = 1 / (1 + np.exp(-a))
    # final activations -> pick max index
    return int(np.argmax(a))

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
# Training Loop with tqdm and time measurements
# -----------------------------------
overall_start_time = time.time()

# Outer loop: generations with a tqdm progress bar
for generation in tqdm(range(generations), unit="generation", ncols=100, dynamic_ncols=True):
    # Reinitialize available images for balanced testing each generation:
    available_images = list(range(min(dataset_sizes.values())))
    available_images.remove(0)
    test_images = {i: [] for i in dataset_sizes.keys()}
    
    # Preload test images (balanced over digits)
    for _ in range(steps // len(dataset_sizes)):
        for i in dataset_sizes:
            if not available_images:
                break
            image_index = random.choice(available_images)
            #available_images.remove(image_index)
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
    os.system("cls")
    print(f"Training Progress: Generation {generation + 1}/{generations}")
    print(f"Best Fitness: {best_fitness}/{total_tests} ({best_fitness/total_tests*100:.2f}%)")
    print(f"Average Fitness: {average_fitness:.2f}/{total_tests} ({average_fitness/total_tests*100:.2f}%)")
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
            remaining_survivors = [s for s in survivors if s is not parent1]
            parent2 = random.choice(remaining_survivors) if remaining_survivors else parent1

            # Average corresponding weight matrices
            child_weights = [ (W1 + W2) / 2.0 for W1, W2 in zip(parent1['weights'], parent2['weights']) ]
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

# --- Save the Best Model (write full layer CSVs for interface.html) ---
model_name = str(input("What would you like to save your model as: "))
model_dir = os.path.join("models", model_name)
os.makedirs(model_dir, exist_ok=True)

# Write each weight matrix as layer_0.csv, layer_1.csv, ...
for idx, W in enumerate(optimal_model['weights']):
    file_path = os.path.join(model_dir, f"layer_{idx}.csv")
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # W is a numpy array
        rows = W.tolist() if hasattr(W, 'tolist') else list(W)
        writer.writerows(rows)

# Final report with total training time and final best model info
total_training_time = time.time() - overall_start_time
os.system("cls")
print("Training Complete!")
print(f"Total Training Time: {total_training_time:.2f}s")
print(f"Best Model Fitness: {optimal_model['fitness']}/{total_tests} ({optimal_model['fitness']/total_tests*100:.2f}%)")
print(f"Model saved in '{model_dir}' successfully!")
