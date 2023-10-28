#pragma once
#include "IConnections.h"
#include <vector>

class NEATConnections :
    public IConnections
{
protected:
    std::vector<size_t> connections_indices;

public:
    void AdjustToNewNeuron(size_t insert_i, bool add_connection) override
    {
        for (size_t i = 0; i < connections_indices.size(); i++)
            connections_indices[i] += connections_indices[i] >= insert_i;

        if (add_connection)
            connections_indices.push_back(insert_i);
    }

    void AdjustToDeletedNeuron(size_t deleted_i) override
    {
        size_t connection_i = -1;
        for (size_t i = 0; i < connections_indices.size() && connection_i != -1; i++)
            connection_i += (i - connection_i) * (connections_indices[i] == deleted_i);

        if (connection_i == -1)
            return;
        
        connections_indices.erase(connections_indices.begin() + connection_i);
    }
};

