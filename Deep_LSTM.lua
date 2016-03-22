--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 16/3/22, 2016.
 - Licence MIT
--]]

local Deep_LSTM = torch.class('Deep_LSTM')

function Deep_LSTM:__init(config)
    self.mem_dim = config.mem_dim or 150
    self.learning_rate = config.learning_rate or 0.05
    self.layers = config.layer or 1
    self.reg = config.reg or 1e-4

    self.emb_dim = config.emb_dim or 300
    self.criterion = nn.ClassNLLCriterion()



end