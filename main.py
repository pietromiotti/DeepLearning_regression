import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
else "cpu")

# Create_Dataset function as given by spec
def create_dataset(sample_size, sigma, w_star, x_range, seed):
    random_state = np.random.RandomState(seed)
    x_min, x_max = x_range
    x = random_state.uniform(x_min, x_max, (sample_size))
    X = np.zeros((sample_size, w_star.shape[0]))
    for i in range(sample_size):
        X[i, 0] = 1
        for j in range(1, w_star.shape[0]):
            X[i, j] = x[i] ** j
    y = X.dot(w_star)
    if sigma > 0:
        y += random_state.normal(0.0, sigma, (sample_size))
    return X, y

#Util function used to create scatterplot
def create_2d_scatter_plot(X, y, motif, show=False):
    plt.scatter(X[:,1], y, marker=motif)
    if(show):
        plt.show()


# General Hyperparameters
w_star = torch.tensor([-8, -4, 2, 1])
x_range = torch.tensor([-3., 2.])
sigma = 0.5
number_of_iterations = 1000
learning_rate = 0.01

# Training Parameters
training_sample_size = 100
training_seed = 0

# Validation Parameters
validation_sample_size = 100
validation_seed = 1

if __name__ == '__main__':

    [X_training_set, Y_training_set] = create_dataset(training_sample_size, sigma, w_star, x_range, training_seed)
    [X_validation_set, Y_validation_set] = create_dataset(validation_sample_size, sigma, w_star, x_range, validation_seed)


    '''
    #Es 3)
    #Plotting code for es 3, uncomment to see the plot

    create_2d_scatter_plot(X_training_set, Y_training_set, motif='o', show=False)
    create_2d_scatter_plot(X_validation_set, Y_validation_set, motif='x', show=False)
    plt.title('Training and Valdaton Set')
    plt.legend(['Traning Data', 'Validation Data'])
    plt.xlabel('Domain')
    plt.ylabel('Value')
    plt.show()
   '''


    X_training_set = torch.tensor(X_training_set, dtype=torch.float32, device=DEVICE)
    Y_training_set = torch.tensor(Y_training_set, dtype=torch.float32, device=DEVICE)
    Y_training_set = Y_training_set.view(Y_training_set.shape[0],1)
    X_validation_set = torch.tensor(X_validation_set, dtype=torch.float32, device=DEVICE)
    Y_validation_set = torch.tensor(Y_validation_set, dtype=torch.float32, device=DEVICE)
    Y_validation_set = Y_validation_set.view(Y_validation_set.shape[0], 1)


    # Set up the model
    model = nn.Linear(X_training_set.shape[1], 1, bias=False)  # input dimension 4, and output dimension 1.

    model = model.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    training_loss_val = torch.zeros(number_of_iterations,1)
    validation_loss_val = torch.zeros(number_of_iterations,1)

    for step in range(number_of_iterations):

        model.train()
        optimizer.zero_grad()
        y_training = model(X_training_set)
        training_loss = loss_fn(y_training, Y_training_set)
        training_loss_val[step] = training_loss.data
        training_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            y_ = model(X_validation_set)
            validation_loss_val[step] = loss_fn(y_, Y_validation_set)


    '''
    #Es 5) Print the weights inferred by the regression, please uncomment it to see the estimated weights
    print("estimated weights", model.weight.data)
    print("real weights", w_star)
    '''

    '''
    #Es 8) Plot the polynomial defined by w_star and the polynomial defined by the model.weights considering 100 points
    between x_range[0] and x_range[1]
    '''
    eval_points = 100
    domain_plot = np.linspace(x_range[0], x_range[1], eval_points)
    polynomial_real = np.zeros(eval_points)
    polynomial_approx = np.zeros(eval_points)
    w_approx = model.weight.data.view(-1)

    for i in range(eval_points):
        x = domain_plot[i]
        polynomial_real[i] = w_star[0] + w_star[1]*x +w_star[2]*(x**2) + w_star[3]*(x**3)
        polynomial_approx[i] = w_approx[0] + w_approx[1] * x + w_approx[2] * (x ** 2) + w_approx[3] * (x ** 3)
    plt.figure()
    plt.plot(domain_plot, polynomial_real)
    plt.plot(domain_plot, polynomial_approx, '--')
    plt.legend(["Polynomial with w*", "Polynomial with w_approx"])
    plt.title('Comparison between Real Polynomial and Estimated one')
    plt.xlabel('Domain')
    plt.ylabel('Value of Polynomial')
    plt.show()


    domain_iteration = np.linspace(0,number_of_iterations,number_of_iterations)
    '''
    Es 7) Plotting the validation and training losses
    '''
    plt.plot(domain_iteration, training_loss_val)
    plt.plot(domain_iteration, validation_loss_val, '--')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Gradient Descent Iterations')
    plt.ylabel('Losses')
    plt.title('Training and Validation Losses comparisons')
    plt.ylim(0,100)
    plt.show()
