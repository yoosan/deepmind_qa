--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 16/3/11, 2016.
 - Licence MIT
--]]

require('torch')
require('sys')
require('lfs')
require('nn')
require('nngraph')
require('optim')
require('xlua')

include('utils/data.lua')
include('model/GRU.lua')
include('model/LSTM.lua')