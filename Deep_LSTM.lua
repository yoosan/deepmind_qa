--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 16/3/22, 2016.
 - Licence MIT
--]]

local Deep_LSTM = torch.class('Deep_LSTM')

function Deep_LSTM:__init(config)
    self.mem_dim = config.mem_dim or 256
    self.learning_rate = config.learning_rate or 0.05
    self.batch_size = config.batch_size or 128
    self.layers = config.layer or 1
    self.vocab_size = config.vocab_size
    self.reg = config.reg or 1e-4
    self.bilstm = config.bilstm or false

    self.emb_dim = config.emb_dim or 256
    self.emb = nn.LookupTable(self.vocab_size, self.emb_dim)
    self.criterion = nn.ClassNLLCriterion()
    self.qa_module = self:new_qa_module()

    self.emb_learning_rate = config.emb_learning_rate or 0.0
    self.optim_state = {
        learning_rate = self.learning_rate,
    }

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

    return nn.gModule(inputs, { logprobs })
end

function Deep_LSTM:train(dataset)
    self.lstm:training()
    self.qa_module:training()
    if self.bilstm then self.lstm_b:training() end

    local indices = torch.randperm(dataset.size)
    for i = 1, dataset.size, self.batch_size do
        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

        local feval = function(x)
            self.grad_params:zero()
            self.emb:zeroGradParameters()

            local loss = 0
            for j = 1, batch_size do
                local idx = indices[i + j - 1]
                local seq, a = dataset:data_iter(dataset.doc_list[idx])
                local inputs = self.emb:forward(seq)
                local rep = self.lstm:forward(inputs)
                local logprobs = self.qa_module:forward(rep)
                local cost_loss = self.criterion:forward(logprobs, a)
                loss = loss + cost_loss
                local cri_grad = self.criterion:backward(logprobs, a)
                local rep_grad = self.qa_module:backward(rep, cri_grad)
                local input_grads
                if self.bilstm then
                    input_grads = self:BiLSTM_backward(seq, inputs, rep_grad)
                else
                    input_grads = self:LSTM_backward(seq, inputs, rep_grad)
                end
                self.emb:backward(seq, input_grads)
            end
            loss = loss / batch_size
            self.grad_params:div(batch_size)
            self.emb.gradWeight:div(batch_size)
            -- regularization
            loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
            self.grad_params:add(self.reg, self.params)
            return loss, self.grad_params
        end
        optim.adagrad(feval, self.params, self.optim_state)
        self.emb:updateParameters(self.emb_learning_rate)
    end
    xlua.progress(dataset.size, dataset.size)
end

-- LSTM backward propagation
function Deep_LSTM:LSTM_backward(seq, inputs, rep_grad)
    local grad
    if self.layers == 1 then
        grad = torch.zeros(seq:nElement(), self.mem_dim)
        grad[seq:nElement()] = rep_grad
    else
        grad = torch.zeros(seq:nElement(), self.layers, self.mem_dim)
        for l = 1, self.layers do
            grad[{ seq:nElement(), l, {} }] = rep_grad[l]
        end
    end
    local input_grads = self.lstm:backward(inputs, grad)
    return input_grads
end

-- Bidirectional LSTM backward propagation
function Deep_LSTM:BiLSTM_backward(seq, inputs, rep_grad)
    local grad, grad_b
    if self.layers == 1 then
        grad = torch.zeros(seq:nElement(), self.mem_dim)
        grad_b = torch.zeros(seq:nElement(), self.mem_dim)
        grad[seq:nElement()] = rep_grad[1]
        grad_b[1] = rep_grad[2]
    else
        grad = torch.zeros(seq:nElement(), self.layers, self.mem_dim)
        grad_b = torch.zeros(seq:nElement(), self.layers, self.mem_dim)
        for l = 1, self.layers do
            grad[{ seq:nElement(), l, {} }] = rep_grad[1][l]
            grad_b[{ 1, l, {} }] = rep_grad[2][l]
        end
    end
    local input_grads = self.lstm:backward(inputs, grad)
    local input_grads_b = self.lstm_b:backward(inputs, grad_b, true)
    return input_grads + input_grads_b
end

function Deep_LSTM:predict(seq)
    self.lstm:evaluate()
    self.qa_module:evaluate()
    local inputs = self.emb:forward(seq)
    local rep
    if self.bilstm then
        self.lstm_b:evaluate()
        rep = {
            self.lstm:forward(inputs),
            self.lstm_b:forward(inputs, true),
        }
    else
        rep = self.lstm:forward(inputs)
    end
    local logprobs = self.qa_module:forward(rep)
    local prediction = utils.argmax(logprobs)

    self.lstm:forget()
    if self.bilstm then self.lstm_b:forget() end
    return prediction

end

function Deep_LSTM:predict_dataset(dataset)
    local predictions = torch.Tensor(dataset.size)
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        predictions[i] = self:predict(dataset:data_iter(dataset.doc_list[i]))
    end
    return predictions
end
