import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import csv
from model import SimpleNN
from pso import ParticleSwarmOptimizer 
# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
#load only from column 0 to column 8
X = dataset[0:614,0:8]
y = dataset[0:614,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

model = SimpleNN()

#print(model)
n_epochs = 1
batch_size = 16
num_of_particles = 100

loss_fn = nn.BCELoss()  # binary cross entropy
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)

lamda = 0.9
weight_decr_step = (0.9 - 0.85) / n_epochs
cognitive_coeff = 1.4845 #Encourages exploitation based on personal best
social_coeff = 1.4845 #Encurages exploitation based on global best
loss_every_epoch = []

print(X.shape[0])
#Check if the number of dimensions of each particle is equal to the number of number of parameters
params_vector = []
for param in model.parameters():
    params_vector.append(param.view(-1))  # Flatten each parameter tensor and append
    
single_vector = torch.cat(params_vector)

#Initialise the optimizer
optimizer  = ParticleSwarmOptimizer(model, loss_fn, num_of_particles, single_vector.shape[0], random.uniform(0.5, 0.8),
                                    cognitive_coeff, social_coeff)
iterations_per_batch = 100
for epoch in range(n_epochs):
    best_in_current_epoch = 9999.0
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y[i:i+batch_size]

        optimizer.optimize(Xbatch, ybatch, iterations_per_batch)
             

#xs = range(0, len(loss_every_epoch))

#plt.plot(np.array(xs), np.array(loss_every_epoch))
#plt.show()
# Make sure to close the plt object once done
#plt.close()
"""
with open('PSO.csv','w') as f:
        writer = csv.writer(f)

        for key in loss_every_epoch:
            writer.writerow([key])

print("Model training done. Checking for accuracy on test set")
"""
X_test =  dataset[615:768,0:8]
Y_test =  dataset[615:768, 8]
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)

with torch.no_grad():

    y_pred_test = optimizer.predict(optimizer.global_best_position, X_test)
    y_pred = optimizer.predict(optimizer.global_best_position, X)

accuracy = (y_pred_test.round() == Y_test).float().mean()
print(f"Accuracy on test set {accuracy}")

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy on training set {accuracy}")

# make class predictions with the model
predictions = ((optimizer.predict(optimizer.global_best_position, X)) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
