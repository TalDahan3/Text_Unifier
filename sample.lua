require 'torch'
require 'nn'

require 'LanguageModel'
local utils = require 'util.utils'

function readAll(file)
    local inFile, err = io.open (file,"r")
    if inFile == nil then
      print("Couldn't open file: "..err)
    else
      local content = inFile:read("*all")
      inFile:close()
      -- Removing EOF
      content = content:sub(1,-3)
      return content
    end
end

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-length', 2000)
cmd:option('-start_text', '')
cmd:option('-start_file', '')
cmd:option('-sample', 1)
cmd:option('-temperature', 1)
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
local opt = cmd:parse(arg)



local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local msg
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  model:cl()
  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
else
  msg = 'Running in CPU mode'
end
--[[T.D read file and copy all the data into a string
    then initialize the start_text parameter with the string ]]
if #opt.start_file > 0 then
  local start_file = opt.start_file
  local text = readAll(start_file)
  cmd:option('-start_text',text)  
  opt = cmd:parse(arg)
end
if opt.verbose == 1 then print(msg) end

model:evaluate()

local sample = model:sample(opt)
print(sample)
