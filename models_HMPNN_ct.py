"""
This file contains the classes that defines the HMPNN_ct-models with various number of layers. 

Main functions: 
    HMPNN_ct_1Layer()
    HMPNN_ct_2Layer()
    HMPNN_ct_3Layer()
    HMPNN_ct_4Layer()
"""

import torch, torch_geometric
from torch_geometric.nn import HeteroConv, NNConv

################################################################################################
def create_dim_in(data, dim):
    """
    Helper function that creates the "dim_in" dictionary used as input to HMPNN_ct_Layer(). 
    The keys of the dict are each node type in the graph, and the value is the number "dim" (same for all keys)
    Input:
        data: The graph object
        dim: An integer holding the representation-dimension for each node-types  (which is the same for all node-types)
    Output: 
        dim_in: A dictionary that maps each node-type in the graph-object (data) to the same integer (dim)
    """
    dim_in = {}
    for node_type in data.node_types: dim_in[node_type] = dim
    return dim_in
################################################################################################

################################################################################################
class HMPNN_ct_Layer(torch.nn.Module):
    """
    This class creates the HMPNN_ct-model for a single node type in a single layer. Assembling multiple objects of this class (that depends on the number of layers in the model and nodes in the graph) will produce a full HMPNN_ct-model. 
    In the case of a single-layer model that makes prediction of a single node-type, this will consist only of a single object of this class. 

    An object of this class takes as input the graph and the node-representation vectors of all nodes of all node-types, and output the new node node-representations for one specific node-type (specified by the input-paramter node_type). 
    The new node-representation is a result of having applied the operation we call "HMPNN with HMCT aggregation".
    This means to perform two main operators: 
        1. For each meta_step ending with node_type, apply the MPNN operator. This will produce one vector for each meta-step, each of which can be seen as an incoming message to nodes of type node_type. 
        2. Concatenate the message-vectors into one vector, and apply a linear transformation to the result. The output is the new representation of the nodes of type node_type

    Input: 
        data: The graph object
        node_type: The node-type to which we will produce new node-representations 
        dim_in: A dictionary holding the node representation-dimension of all node types (before applying this layer). If the operator is created for the first layer of the full model, this is the (original) number of node-features for each node type. If it is not in the first layer, it is the representation-dimension  of each node-types in the previous layer (if so, the current approach is that all node_types will have the same representation-dimension). If dim_in is not provided, the class assumes that it is the first layer and sets dim_in to correspond to the number of features for each node-type.
        dim_message: Determines the dimension of the vectors that is the output accross each meta-step (i.e. for each MPNN-operator). This dimension is the same for all meta-steps. 
        dim_out: The dimension of the new node-representation for nodes of type node_type (the dimension of the output produced by this operator) 

    (Additional) Class Variables: 
        num_meta_steps: The number of meta-steps incoming to node_type
        message_passers: List of the MPNN-operators. Has the length num_meta_steps (one for each incoming meta-step)
        linear: THe linear layer that transforms the concatenated output-vectors from message_passers to the final output, which is the new representation of node_type (has dimension dim_out)

    """
    def __init__(self
            , data: torch_geometric.data.hetero_data.HeteroData
            , node_type: str = "indivi"
            , dim_in: dict = {}     
            , dim_message: int = -1
            , dim_out: int = 1
            ):
        super().__init__()

        # Initialize node_type, dim_in, dim_message, dim_out and num_meta_steps
        self.node_type = node_type
        self.dim_in = dim_in
        if len(self.dim_in) == 0:
            for nt in data.node_types: self.dim_in[nt] = data[nt].x.shape[1] # If dim_in is not provided, we set it equal to the feature-dimension of each node_type
        self.dim_message = dim_message
        if self.dim_message == -1: self.dim_message = data[node_type].x.shape[1] # if dim_message is not given, we set it equal to the number of features of the node_type that will receive messages 
        self.dim_out = dim_out
        self.num_meta_steps = sum([meta_step[2]==node_type for meta_step in data.edge_types]) # num_meta_steps holds the number of meta-steps incoming to node_type

        # Initialize the MPNN-operators for each meta-step ending in node_type, and placing them in a list
        self.message_passers = torch.nn.ModuleList()
        for meta_step in data.edge_types:
            if meta_step[2] == self.node_type:
                num_edge_features = data[meta_step].edge_attr.shape[1]
                message_function = torch.nn.Linear(num_edge_features, self.dim_in[meta_step[0]]*self.dim_message)
                mp = {}
                mp[meta_step] = NNConv((self.dim_in[meta_step[0]], self.dim_in[meta_step[2]]), self.dim_message, message_function)
                self.message_passers.append(HeteroConv(mp, aggr = "sum" ))

        # Initialize the linear layer that the concatenated message vectors will be transformed by
        self.linear = torch.nn.Linear(len(self.message_passers)*self.dim_message, self.dim_out)

    def print_data(self):
        # A helper-function to inspect the value of the class-variables
        print(f'node_type: {self.node_type}')
        print(f'dim_in: {self.dim_in}')
        print(f'dim_message: {self.dim_message}')
        print(f'dim_out: {self.dim_out}')
        print(f'num_meta_steps: {self.num_meta_steps}')
        print(f'message_passers: {self.message_passers}')
        print(f'linear: {self.linear}')

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Apply message passing accross each meta-step and concatenate the results into a single tensor
        for i in range(self.num_meta_steps): 
            msg = self.message_passers[i](x_dict, edge_index_dict, edge_attr_dict)[self.node_type]
            msg = torch.sigmoid(msg)
            if "msg_cat" not in locals(): msg_cat = msg
            else: msg_cat = torch.cat((msg_cat, msg), 1)
        # Apply linear transformation, followed by the sigmoid function, and return
        return torch.sigmoid(self.linear(msg_cat))
################################################################################################


################################################################################################
class HMPNN_ct_1Layer(torch.nn.Module):
    """
    The class that defines the 1-layer HMPNN_ct-model used for binary node-classification on nodes of type node_type 
    Input: 
        data: The graph object
        node_type: The node_type to classify (predict which binary class each node of type node_type belongs to)
    """
    def __init__(self
            , data: torch_geometric.data.hetero_data.HeteroData
            , node_type: str = "indivi"   
            ):
        super().__init__()

        self.node_type = node_type 
        self.dim_message_layer_1 = 10 # Determines the dimension of the vectors that is the output accross each meta-step (i.e. for each MPNN-operator). This dimension is the same for all meta-steps. 
        self.dim_out = 1 # Specifies the final dimension of the representation vector for node_type (the final output of the class). Since this is used for binary classification, it is set to 1. 

        # Creating the HMPNN_ct-operator for the node-type node_type for the single-layer model. 
        self.layer = HMPNN_ct_Layer(data, node_type = self.node_type, dim_message=self.dim_message_layer_1, dim_out = self.dim_out)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Applying the model
        return self.layer(x_dict, edge_index_dict, edge_attr_dict)
################################################################################################

################################################################################################
class HMPNN_ct_2Layer(torch.nn.Module):
    """
    The class that defines the 2-layer HMPNN_ct-model used for binary node-classification on nodes of type node_type 
    Input: 
        data: The graph object
        node_type: The node_type to classify (predict which binary class each node of type node_type belongs to)
    """
    def __init__(self
            , data: torch_geometric.data.hetero_data.HeteroData
            , node_type: str = "indivi"     
            ):
        super().__init__()

        self.node_type = node_type 
        self.node_types = data.node_types # All node types in the graph

        self.dim_message_layer_1 = 2 # Choice from thesis: 2
        self.dim_message_layer_2 = 10 # Choice from thesis (final layer): 10
        self.dim_out_layer_1 = 5 # Choice from thesis: 5
        self.dim_out = 1 # Specifies the final dimension of the representation vector for node_type (the final output of the class). Since this is used for binary classification, it is set to 1. 

        # dim_in_2 is used as input when creating the second layer "hmct message passing operator". 
        #   The variable is used to specify the representation/feature dimension of each node type in the graph after having been transformed by the first layer. This is required because we allow the representation dimension for a node type change from the dimension of its original feature vector after a message passing layer is applied. 
        self.dim_in_2 = create_dim_in(data, self.dim_out_layer_1) 

        # Creating the HMPNN_ct-operators for the first layer: one operator for each node_type in the graph. 
        self.layer_1 = torch.nn.ModuleList()
        for nt in data.node_types:
            self.layer_1.append(HMPNN_ct_Layer(data, node_type = nt, dim_message=self.dim_message_layer_1, dim_out = self.dim_out_layer_1))

        # Creating the HMPNN_ct-operators for the output-layer: Only created for node_type (the node to make predictions on)
        self.layer_2 = HMPNN_ct_Layer(data, node_type = node_type, dim_in = self.dim_in_2, dim_message=self.dim_message_layer_2, dim_out = self.dim_out)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict_updates = {}
        # Apply layer 1
        for node_update_fun in self.layer_1:
            x_dict_updates[node_update_fun.node_type] = node_update_fun.forward(x_dict, edge_index_dict, edge_attr_dict)
        # Apply output layer
        return self.layer_2.forward(x_dict_updates, edge_index_dict, edge_attr_dict)
################################################################################################

################################################################################################
class HMPNN_ct_3Layer(torch.nn.Module):
    """
    The class that defines the 3-layer HMPNN_ct-model used for binary node-classification on nodes of type node_type 
    Input: 
        data: The graph object
        node_type: The node_type to classify (predict which binary class each node of type node_type belongs to)
    """
    def __init__(self
            , data: torch_geometric.data.hetero_data.HeteroData
            , node_type: str = "indivi"
            ):
        super().__init__()

        self.node_type = node_type 
        self.node_types = data.node_types # All node types in the graph

        self.dim_message_layer_1 = 5 # Choice from thesis: 2
        self.dim_message_layer_2 = 5 # Choice from thesis: 2
        self.dim_message_layer_3 = 15 # Choice from thesis (final layer): 10

        self.dim_out_layer_1 = 10 # Choice from thesis: 5
        self.dim_out_layer_2 = 10 # Choice from thesis: 5
        self.dim_out_layer_3 = 1 # Specifies the final dimension of the representation vector for node_type (the final output of the class)

        # dim_in_2 is used as input when creating the second layer "hmct message passing operator". 
        #   The variable is used to specify the representation/feature dimension of each node type in the graph after having been transformed by the first layer. This is required because we allow the representation dimension for a node type to change from the dimension of its original feature vector after a message passing layer is applied. 
        self.dim_in_2 = create_dim_in(data, self.dim_out_layer_1) 
        self.dim_in_3 = create_dim_in(data, self.dim_out_layer_2) 

      # Creating the HMPNN_ct-operators for the first and second layer: one operator for each node_type in the graph, for each layer. 
        self.layer_1 = torch.nn.ModuleList()
        self.layer_2 = torch.nn.ModuleList()
        for nt in data.node_types:
            self.layer_1.append(HMPNN_ct_Layer(data, node_type = nt, dim_message=self.dim_message_layer_1, dim_out = self.dim_out_layer_1))
            self.layer_2.append(HMPNN_ct_Layer(data, node_type = nt, dim_in = self.dim_in_2, dim_message=self.dim_message_layer_2, dim_out = self.dim_out_layer_2))

      # Creating the HMPNN_ct-operators for the output-layer: Only created for node_type (the node to make predictions on)
        self.layer_3 = HMPNN_ct_Layer(data, node_type = node_type, dim_in = self.dim_in_3, dim_message=self.dim_message_layer_3, dim_out = self.dim_out_layer_3)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):        
        x_dict_updates = {}
        # Apply layer 1
        for node_update_fun in self.layer_1:
            x_dict_updates[node_update_fun.node_type] = node_update_fun.forward(x_dict, edge_index_dict, edge_attr_dict)
        # Apply layer 2
        for node_update_fun in self.layer_2:
            x_dict_updates[node_update_fun.node_type] = node_update_fun.forward(x_dict_updates, edge_index_dict, edge_attr_dict)
        # Apply output layer
        return self.layer_3.forward(x_dict_updates, edge_index_dict, edge_attr_dict)
################################################################################################

################################################################################################
class HMPNN_ct_4Layer(torch.nn.Module):
    """
    The class that defines the 4-layer HMPNN_ct-model used for binary node-classification on nodes of type node_type 
    Input: 
        data: The graph object
        node_type: The node_type to classify (predict which binary class each node of type node_type belongs to)
    """
    def __init__(self
            , data: torch_geometric.data.hetero_data.HeteroData
            , node_type: str = "indivi"
            ):
        super().__init__()

        self.node_type = node_type 
        self.node_types = data.node_types # All node types in the graph

        self.dim_message_layer_1 = 2 # Choice from thesis: 2
        self.dim_message_layer_2 = 2 # Choice from thesis: 2
        self.dim_message_layer_3 = 2 # Choice from thesis: 2
        self.dim_message_layer_4 = 10 # Choice from thesis (final layer): 10

        self.dim_out_layer_1 = 5 # Choice from thesis: 5
        self.dim_out_layer_2 = 5 # Choice from thesis: 5
        self.dim_out_layer_3 = 5 # Choice from thesis: 5
        self.dim_out_layer_4 = 1 # Specifies the final dimension of the representation vector for node_type (the final output of the class)

        # dim_in_2 is used as input when creating the second layer "hmct message passing operator". 
        #   The variable is used to specify the representation/feature dimension of each node type in the graph after having been transformed by the first layer. This is required because we allow the representation dimension for a node type to change from the dimension of its original feature vector after a message passing layer is applied. 
        self.dim_in_2 = create_dim_in(data, self.dim_out_layer_1) 
        self.dim_in_3 = create_dim_in(data, self.dim_out_layer_2) 
        self.dim_in_4 = create_dim_in(data, self.dim_out_layer_3) 

      # Creating the HMPNN_ct-operators for the first, second and third layer: one operator for each node_type in the graph, for each layer. 
        self.layer_1 = torch.nn.ModuleList()
        self.layer_2 = torch.nn.ModuleList()
        self.layer_3 = torch.nn.ModuleList()
        for nt in data.node_types:
            self.layer_1.append(HMPNN_ct_Layer(data, node_type = nt, dim_message=self.dim_message_layer_1, dim_out = self.dim_out_layer_1))
            self.layer_2.append(HMPNN_ct_Layer(data, node_type = nt, dim_in = self.dim_in_2, dim_message=self.dim_message_layer_2, dim_out = self.dim_out_layer_2))
            self.layer_3.append(HMPNN_ct_Layer(data, node_type = nt, dim_in = self.dim_in_3, dim_message=self.dim_message_layer_3, dim_out = self.dim_out_layer_3))

      # Creating the HMPNN_ct-operators for the output-layer: Only created for node_type (the node to make predictions on)
        self.layer_4 = HMPNN_ct_Layer(data, node_type = node_type, dim_in = self.dim_in_4, dim_message=self.dim_message_layer_4, dim_out = self.dim_out_layer_4)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):        
        x_dict_updates = {}
        # Apply layer 1
        for node_update_fun in self.layer_1:
            x_dict_updates[node_update_fun.node_type] = node_update_fun.forward(x_dict, edge_index_dict, edge_attr_dict)
        # Apply layer 2
        for node_update_fun in self.layer_2:
            x_dict_updates[node_update_fun.node_type] = node_update_fun.forward(x_dict_updates, edge_index_dict, edge_attr_dict)
        # Apply layer 3
        for node_update_fun in self.layer_3:
            x_dict_updates[node_update_fun.node_type] = node_update_fun.forward(x_dict_updates, edge_index_dict, edge_attr_dict)
        # Apply output layer
        return self.layer_4.forward(x_dict_updates, edge_index_dict, edge_attr_dict)
################################################################################################