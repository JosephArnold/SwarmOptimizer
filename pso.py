import torch
import numpy as np

class ParticleSwarmOptimizer:
    def __init__(self, model, loss_function, num_particles, 
                 dimensions, inertia=0.5, cognitive_factor=1.5, 
                 social_factor=1.5):
        """
        Initialize the PSO class.
        
        Args:
            model (torch.nn.Module): The neural network model to optimize.
            num_particles (int): Number of particles in the swarm.
            dimensions (int): The number of dimensions (same as number of model parameters).
            w (float): Inertia weight.
            c1 (float): Cognitive (self-attraction) weight.
            c2 (float): Social (global-attraction) weight.
        """
        self.model = model
        self.loss_function = loss_function
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.inertia = inertia
        self.cognitive_factor = cognitive_factor
        self.social_factor = social_factor

        # Initialize particles' positions and velocities
        self.particle_positions = torch.rand(num_particles, dimensions)  # Random initial positions
        self.particle_velocities = torch.rand(num_particles, dimensions) * 0.1  # Small initial velocities
        
        # Initialize personal bests and global best
        self.personal_best_positions = self.particle_positions.clone()
        self.personal_best_scores = torch.full((num_particles,), float('inf'))
        self.global_best_position = torch.zeros(dimensions)
        self.global_best_score = float('inf')

    def predict(self, particle_position, input):
        
        # Update the model parameters based on the particle's position
        start = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = particle_position[start:start + param_size].view(param.size())
            start += param_size

        predicted = self.model(input)
        return predicted

    def fitness_function(self, particle_position, input, expected):
        """Calculate the fitness of a particle (model loss)."""
        
        predicted = self.predict(particle_position, input)
        loss = self.loss_function(predicted, expected)
        return loss

    def update_particles(self, input, expected):
        """Update particle velocities and positions based on the best-known positions."""
        for i in range(self.num_particles):
            # Evaluate fitness
            fitness = self.fitness_function(self.particle_positions[i], input, expected)
            
            # Update personal best
            if fitness < self.personal_best_scores[i]:
                self.personal_best_scores[i] = fitness
                self.personal_best_positions[i] = self.particle_positions[i].clone()
            
            # Update global best
            if fitness < self.global_best_score:
                self.global_best_score = fitness
                self.global_best_position = self.particle_positions[i].clone()

        # Update velocities and positions
        for i in range(self.num_particles):
            r1 = torch.rand(self.dimensions)
            r2 = torch.rand(self.dimensions)
            
            cognitive_velocity = self.cognitive_factor * r1 * (self.personal_best_positions[i] - self.particle_positions[i])
            social_velocity = self.social_factor * r2 * (self.global_best_position - self.particle_positions[i])
            self.particle_velocities[i] = self.inertia * self.particle_velocities[i] + cognitive_velocity + social_velocity
            self.particle_positions[i] += self.particle_velocities[i]

    def optimize(self, input, expected, iterations):
        """Run the PSO optimization for a number of iterations."""
        for iteration in range(iterations):
            self.update_particles(input, expected)
            #print(f"Iteration {iteration + 1}/{iterations}, Global Best Score: {self.global_best_score}")
        print(f"Global Best Score: {self.global_best_score}")        
        return self.global_best_position, self.global_best_score

