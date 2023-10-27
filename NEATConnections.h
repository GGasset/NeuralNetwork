#pragma once
#include "IConnections.h"
#include <vector>

class NEATConnections :
    public IConnections
{
protected:
    std::vector<size_t> connections_indices;
};

