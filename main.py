import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import csv
from model import SimpleNN

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
#load only from column 0 to column 8
X = dataset[0:614,0:8]

#load column 8 which is the output vector
y = dataset[0:614,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


model = SimpleNN()

#print(model)
n_epochs = 25
batch_size = 16
n_input_layer = 8
n_layer1 = 12
n_layer2 = 8
n_output_layer = 1
num_of_particles = 200
dimensions = 221

def updateParameters(current_weight):
    base = 0    
    for param in model.parameters():
    # Get the number of elements in the current parameter
        num_params = param.numel()

        # Extract the corresponding portion from the new_params_vector
        new_values =  curr_weight[base:base + num_params]

        # Reshape the values to the parameter's shape and assign them
        param.data = new_values.view(param.size())

        # Update the start index for the next layer
        base += num_params

loss_fn = nn.BCELoss()  # binary cross entropy
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)

#initialise particles in the swarm with random positions
#Choose an arbitary number of particles 'N'= 10
#Each particle will have the following number of dimensions = (12 * 8) + (8 * 12) + (1 * 8) + 12 + 8 + 1 = 221

#initialize particle's positions and initial velocity 'vt' randomly
particle_positions = torch.rand(num_of_particles, dimensions) * 0.01
vt =  torch.rand(num_of_particles, dimensions) * 10
vt_next = torch.rand(num_of_particles, dimensions)

#initialize global and particle best positions
global_best_position = torch.zeros(1, dimensions)
particle_best_positions = torch.zeros(num_of_particles, dimensions)
global_best_error =  9999.0
particle_best_error = torch.full((num_of_particles, 1), 9999.0)

#initialize scalar values of inertia weight 'w', cognitive weight 'c1', global weight 'c2' and random factors 'r1' and 'r2'
#w = random.uniform(0.6, 0.9)
lamda = 0.9
weight_decr_step = (0.9 - 0.85) / n_epochs
cognitive_coeff = 1.4845 #Encourages exploitation based on personal best
social_coeff = 1.4845 #Encurages exploitation based on global best
#c2 = random.uniform(0, 1)
loss_every_epoch = []

print(X.shape[0])
#Check if the number of dimensions of each particle is equal to the number of number of parameters
params_vector = []
for param in model.parameters():
    params_vector.append(param.view(-1))  # Flatten each parameter tensor and append
    
single_vector = torch.cat(params_vector)


assert  single_vector.shape[0] == particle_positions.shape[1], "Number of dimensions in Particle != Number of parameters"
        
for epoch in range(n_epochs):
    best_in_current_epoch = 9999.0
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y[i:i+batch_size]


        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        w =  random.uniform(0.5, 0.8)

        #Check if the number of dimensions of each particle is equal to the number of number of parameters 
        #run forward prop for each particle
        for j in range(0, num_of_particles): 
            
            #update weights of particles
            #get position of particle(i) and store it in model weight and biases
            curr_weight = particle_positions[j]
            
            #update model parameters with particle[j]
            updateParameters(curr_weight)
            
            y_pred = model(Xbatch) #predict using current particle position as weights
            loss = loss_fn(y_pred, ybatch) #calculate loss
             
        #    Use PSO to optimize the weights here. Reference given below. 
        #    https://pyswarms.readthedocs.io/en/latest/examples/usecases/train_neural_network.html
        #    1. Update particle best position and global best position based on error
        #    2. Update velocity
        #       v(t+1) = (w * v(t)) + (c1 * r1 * (p(t) – x(t)) + (c2 * r2 * (g(t) – x(t))
        #    3. Update position

            
            #print("velocity of particles before ")
            #print(vt)

            if(loss < best_in_current_epoch):
                best_in_current_epoch = loss.item()
            if(loss < particle_best_error[j]):
                particle_best_error[j] = loss
                particle_best_positions[j] = curr_weight
            if(loss < global_best_error):
                global_best_error = loss
                global_best_position = curr_weight
            
            #vt_next[j] = (w * vt[j] + ((c1 * r1 * (particle_best_positions[j] - curr_weight)) + (c2 * r2 * (global_best_position - curr_weight)))) * (1 - lamda) + lamda * vt[j]
            vt_next[j] = (w * vt[j] + ((cognitive_coeff * r1 * (particle_best_positions[j] - curr_weight)) + (social_coeff * r2 * (global_best_position - curr_weight)))) 
            particle_positions[j] = torch.add(particle_positions[j] , vt_next[j])      

            vt[j] = vt_next[j]
            

    #w -= weight_decr_step
    loss_every_epoch = loss_every_epoch + [best_in_current_epoch]
    print(f'Finished epoch {epoch}, latest loss {global_best_error} lowest loss this epoch {best_in_current_epoch}')

xs = range(0, len(loss_every_epoch))

plt.plot(np.array(xs), np.array(loss_every_epoch))
plt.show()
# Make sure to close the plt object once done
plt.close()

with open('PSO.csv','w') as f:
        writer = csv.writer(f)

        for key in loss_every_epoch:
            writer.writerow([key])

print("Model training done. Checking for accuracy on test set")

X_test =  dataset[615:768,0:8]
Y_test =  dataset[615:768, 8]
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)

with torch.no_grad():

    updateParameters(global_best_position)
 
    y_pred_test = model(X_test)
    y_pred = model(X)

accuracy = (y_pred_test.round() == Y_test).float().mean()
print(f"Accuracy on test set {accuracy}")

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy on training set {accuracy}")


# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

