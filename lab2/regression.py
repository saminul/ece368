import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    steps = 150   
    x_values = np.linspace(-1, 1, steps)
    y_values = np.linspace(-1, 1, steps)

    X, Y = np.meshgrid(x_values, y_values)
    x_set = []
    for col in y_values:
        for row in x_values:
            x_set.append([row,col])
    x_set = np.asarray(x_set)

    mean = np.array([0, 0])
    cov = np.array([[beta, 0], [0, beta]])

    density = util.density_Gaussian(mean, cov, x_set)
    # print(density.shape)
    Z = density.reshape((steps,steps))
    # print(Z.shape)

    plt.figure(1)
    plt.title("Prior Distribution-P(a)")
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.contour(X, Y, Z)
    plt.plot(-0.1, -0.5, 'bx', label='a true value')
    plt.legend()
    plt.show()
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    #
    col = np.ones((x.shape[0], 1))
    x = np.append(col, x, axis = 1)

    intermediate = x.T@x + (sigma2/beta)*np.identity(2, dtype=float)
    mu = (np.linalg.inv(intermediate)@x.T@z).flatten()
    Cov = np.linalg.inv(intermediate)*sigma2

    steps = 200   
    x_values = np.linspace(-1, 1, steps)
    y_values = np.linspace(-1, 1, steps)

    X, Y = np.meshgrid(x_values, y_values)

    x_set = []
    for col in y_values:
        for row in x_values:
            x_set.append([row,col])
    x_set = np.asarray(x_set)

    density = util.density_Gaussian(mu, Cov, x_set)
    Z = density.reshape((steps,steps))

    plt.figure(2)
    plt.title("Posterior distrubition for " + str(len(x)) + " samples")
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.contour(X, Y, Z)
    plt.plot(-0.1, -0.5, 'bx', label='a true value')
    plt.legend()
    plt.show()

    return (mu,Cov)
    


def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    X_new = np.column_stack((np.ones((len(x), 1)), np.array(x)))

    deviation = np.sqrt(np.diag(sigma2 + ((X_new @ Cov) @ X_new.T)))
    mu_new = X_new @ mu

    plt.figure(3)
    plt.title('Regression Result for {} training samples'.format(x_train.shape[0]))
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title("Prediction for " + str(x_train.shape[0]) + " samples")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.errorbar(np.array(x), mu_new, yerr=deviation, ecolor='k', color='r', label='Predictions')
    plt.scatter(x_train, z_train, color = 'b', label='Training Samples')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)

