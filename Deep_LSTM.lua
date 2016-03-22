--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 16/3/22, 2016.
 - Licence MIT
--]]

local Deep_LSTM = torch.class('Deep_LSTM')

function Deep_LSTM:__init(config)
    self.mem_dim = config.mem_dim or 256
    self.learning_rate = config.learning_rate or 0.05
    self.layers = config.layer or 1
    self.vocab_size = config.vocab_size
    self.reg = config.reg or 1e-4
    self.bilstm = config.bilstm or false

    self.emb_dim = config.emb_dim or 256
    self.criterion = nn.ClassNLLCriterion()
    self.qa_module = self:new_qa_module()

    local lstm_config = {
        in_dim = self.emb_dim,
        mem_dim = self.mem_dim,
        layers = self.layers,
        gate_output = true,
    }
    self.lstm = nn.LSTM(lstm_config)

    if self.bilstm then
        self.lstm_b = nn.LSTM(lstm_config)
        utils.share_params(self.lstm, self.bilstm)
    end
    local modules = nn.Parallel():add(self.lstm):add(self.qa_module)
    self.params, self.grad_params = modules:getParameters()
end

function Deep_LSTM:new_qa_module()
    local input_dim = self.layers * self.mem_dim
    local inputs, vec
    if self.bilstm then
        local frep, brep = nn.Identity()(), nn.Identity()()
        input_dim = input_dim * 2
        if self.layers == 1 then
            vec = nn.JoinTable(1) { frep, brep }
        else
            vec = nn.JoinTable(1) { nn.JoinTable(1)(frep), nn.JoinTable(1)(brep) }
        end
        inputs = { frep, brep }
    else
        local rep = nn.Identity()()
        if self.layers == 1 then
            vec = { rep }
        else
            vec = nn.JoinTable(1)(rep)
        end
        inputs = { rep }
    end

    local logprobs
    if self.dropout then
        logprobs = nn.LogSoftMax()(nn.Linear(input_dim, self.vocab_size)(nn.Dropout()(vec)))
    else
        logprobs = nn.LogSoftMax()(nn.Linear(input_dim, self.vocab_size)(vec))
    end

    return nn.gMoudle(inputs, { logprobs })
end

function Deep_LSTM:train()
end