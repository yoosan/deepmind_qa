--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 16/3/21, 2016.
 - Licence MIT
--]]

utils = {}
function utils.contain(value, container)
    for _, v in container do
        if v == value then
            return true
        end
    end
    return false
end

function utils.share_params(cell, src)
    if torch.type(cell) == 'nn.gModule' then
        for i = 1, #cell.forwardnodes do
            local node = cell.forwardnodes[i]
            if node.data.module then
                node.data.module:share(src.forwardnodes[i].data.module,
                    'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    elseif torch.isTypeOf(cell, 'nn.Module') then
        cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
    else
        error('parameters cannot be shared for this input')
    end
end