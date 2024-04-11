"""
This file contains the classes that defines the HMPNN models with sum as aggregation and with various number of layers. 
These models are not intended to be used as final models, but are used to benchmark the HMPNN_ct-models. 

Main functions: 
    HMPNN_sum_1Layer()
    HMPNN_sum_2Layer()
    HMPNN_sum_3Layer()
    HMPNN_sum_4Layer()

"""

import torch
from torch_geometric.nn import HeteroConv, NNConv


class OutputLayer(torch.nn.Module):
    def __init__(self, data, node_type="indivi"):
        super().__init__()

        self.node_type = node_type

        # We iterate over all meta-paths in the graph, and create a message passing function for each one where the end-node type is "nt".
        # We place all the message passing functions into mp_dict
        mp_dict = {}
        for meta_step in data.metadata()[1]:
            if meta_step[2] == self.node_type:
                num_node_features_outgoing = data[meta_step[0]].x.shape[1]
                num_node_features_incoming = data[meta_step[2]].x.shape[1]
                num_edge_features = data[meta_step].edge_attr.shape[1]
                num_features_new = 1

                message_function = torch.nn.Linear(
                    num_edge_features, num_node_features_outgoing * num_features_new
                )
                mp = NNConv(
                    (num_node_features_outgoing, num_node_features_incoming),
                    num_features_new,
                    message_function,
                    aggr="add",
                )
                mp_dict[meta_step] = mp
        # Create the heteroconv using the dictionary of message passing functions
        self.conv = HeteroConv(mp_dict, aggr="sum")

    def print_self(self):
        print("MESSAGE PASSINGS")
        print(getattr(getattr(model, "conv"), "convs"))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        return self.conv(x_dict, edge_index_dict, edge_attr_dict)


class InnerLayer(torch.nn.Module):
    def __init__(self, data):
        super().__init__()

        mp_dict = {}
        for meta_step in data.metadata()[1]:
            num_node_features_outgoing = data[meta_step[0]].x.shape[1]
            num_node_features_incoming = data[meta_step[2]].x.shape[1]
            num_edge_features = data[meta_step].edge_attr.shape[1]

            message_function = torch.nn.Linear(
                num_edge_features,
                num_node_features_outgoing * num_node_features_incoming,
            )
            mp = NNConv(
                (num_node_features_outgoing, num_node_features_incoming),
                num_node_features_incoming,
                message_function,
                aggr="add",
            )
            mp_dict[meta_step] = mp

        self.conv = HeteroConv(mp_dict, aggr="sum")

    def print_self(self):
        print("MESSAGE PASSINGS")
        print(getattr(getattr(model, "conv"), "convs"))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        return self.conv(x_dict, edge_index_dict, edge_attr_dict)


# One Layer NNConv
class HMPNN_sum_1Layer(torch.nn.Module):
    def __init__(self, data, node_type="indivi"):
        super().__init__()
        self.node_type = node_type

        self.conv = OutputLayer(data, node_type=self.node_type)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)
        return torch.sigmoid(x_dict[self.node_type])


class HMPNN_sum_2Layer(torch.nn.Module):
    def __init__(self, data, node_type="indivi"):
        super().__init__()
        self.node_type = node_type

        self.conv1 = InnerLayer(data)
        self.conv2 = OutputLayer(data, node_type=self.node_type)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict_updates = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        for node_type in x_dict_updates.keys():
            x_dict[node_type] = torch.sigmoid(x_dict_updates[node_type])
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        return torch.sigmoid(x_dict[self.node_type])


class HMPNN_sum_3Layer(torch.nn.Module):
    def __init__(self, data, node_type="indivi"):
        super().__init__()
        self.node_type = node_type

        self.conv1 = InnerLayer(data)
        self.conv2 = InnerLayer(data)
        self.conv3 = OutputLayer(data, node_type=self.node_type)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict_updates = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        for node_type in x_dict_updates.keys():
            x_dict[node_type] = torch.sigmoid(x_dict_updates[node_type])
        x_dict_updates = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        for node_type in x_dict_updates.keys():
            x_dict[node_type] = torch.sigmoid(x_dict_updates[node_type])
        x_dict = self.conv3(x_dict, edge_index_dict, edge_attr_dict)
        return torch.sigmoid(x_dict[self.node_type])


class HMPNN_sum_4Layer(torch.nn.Module):
    def __init__(self, data, node_type="indivi"):
        super().__init__()
        self.node_type = node_type

        self.conv1 = InnerLayer(data)
        self.conv2 = InnerLayer(data)
        self.conv3 = InnerLayer(data)
        self.conv4 = OutputLayer(data, node_type=self.node_type)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Layer 1
        x_dict_updates = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        for node_type in x_dict_updates.keys():
            x_dict[node_type] = torch.sigmoid(x_dict_updates[node_type])
        # Layer 2
        x_dict_updates = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        for node_type in x_dict_updates.keys():
            x_dict[node_type] = torch.sigmoid(x_dict_updates[node_type])
        # Layer 3
        x_dict_updates = self.conv3(x_dict, edge_index_dict, edge_attr_dict)
        for node_type in x_dict_updates.keys():
            x_dict[node_type] = torch.sigmoid(x_dict_updates[node_type])
        # Output layer
        x_dict = self.conv4(x_dict, edge_index_dict, edge_attr_dict)
        return torch.sigmoid(x_dict[self.node_type])
