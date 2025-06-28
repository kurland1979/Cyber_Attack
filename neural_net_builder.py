import torch.nn as nn

class NeuralNetBuilder(nn.Module):
    # Flexible feedforward neural network with configurable hidden layers.
    
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        layers = []
        in_dim = input_dim  
        for hidden_dim in hidden_layers:
            #  Initialize the neural network structure.
            """
            Parameters
            ----------
            input_dim : int
            Number of input features.
            output_dim : int
            Number of output classes/targets.
            hidden_layers : list of int
            List specifying the number of units in each hidden layer.
            """
            layers.append(nn.Linear(in_dim, hidden_dim))      
            layers.append(nn.BatchNorm1d(hidden_dim))         
            layers.append(nn.ReLU())                          
            layers.append(nn.Dropout(0.3))                    
            in_dim = hidden_dim                               
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        # Forward pass of the network.
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        
        """
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


    
       
     





