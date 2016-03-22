--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 16/3/11, 2016.
 - Licence MIT
--]]

local RCDataset = torch.class('RCDataset')

function RCDataset:__init(config)
    self.sources = { 'context', 'question', 'answer', 'candidates' }
    self.dataset_file = config.dataset_file
    self.vocab_file = config.vocab_file
    self.n_entities = config.n_entities
    self.vocab, self.ivocab = self:build_vocab()
    self.vocab_size = #(self.vocab)
end

function RCDataset:build_vocab()
    local vocab_file = torch.DiskFile(self.vocab_file)
    local vocab = vocab_file:readString('*a'):split('\n')
    local ivocab = {}
    for i = 1, self.n_entities do
        table.insert(vocab, string.format('@entity%d', i - 1))
    end
    table.insert(vocab, '<UNK>')
    table.insert(vocab, '@placeholder')
    table.insert(vocab, '<SEP>')
    for k, v in ipairs(vocab) do
        ivocab[v] = k
    end
    return vocab, ivocab
end

function RCDataset:to_word_idx(w, cand_map)
    if cand_map[w] ~= nil then
        return cand_map[w]
    elseif self.ivocab[w] ~= nil then
        return self.ivocab[w]
    else
        return self.ivocab['<UNK>']
    end
end

function RCDataset:to_word_ids(s, cand_map)
    local sents = s:split(' ')
    local sent_ids = {}
    for i = 1, #sents do
        table.insert(sent_ids, self:to_word_idx(sents[i], cand_map))
    end
    return torch.Tensor(sent_ids)
end

function RCDataset:data_iter(doc_iter)
    local data_file = torch.DiskFile(self.dataset_file .. doc_iter)
    local lines = data_file:readString('*a'):split('\n')
    local ctx, q, a, cands = lines[3], lines[5], lines[7], {}
    for i = 8, #lines do
        local line = lines[i]
        local cand = line:split(':')[1]
        table.insert(cands, self:to_word_idx(cand, {}))
    end

    if self.n_entities >= #cands then
        ctx = self:to_word_ids(ctx, {})
        q = self:to_word_ids(q, {})
        a = self:to_word_ids(a, {})
        cands = torch.Tensor(cands)
        return { ['context'] = ctx, ['question'] = q, ['answer'] = a, ['candidates'] = cands }
    else
        print('[*] error -> please set n_entities >= #cands')
        return
    end
end