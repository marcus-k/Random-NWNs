"""


"""
function get_voltages(NWN_dict, V, edge_list)
    edge_num = length(edge_list)
    v0, v1 = zeros(edge_num), zeros(edge_num)
    for (i, edge) in enumerate(edge_list)
        # Julia uses 1-based indexing
        v0_indx = NWN_dict[edge[1]]
        v1_indx = NWN_dict[edge[2]]
        v0[i] = V[v0_indx + 1]
        v1[i] = V[v1_indx + 1]
    end

    v0, v1
end
