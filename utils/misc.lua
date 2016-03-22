--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 16/3/21, 2016.
 - Licence MIT
--]]

local utils = {}
function utils.contain(value, container)
    for _, v in container do
        if v == value then
            return true
        end
    end
    return false
end
