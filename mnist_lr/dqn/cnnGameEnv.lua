require 'cunn'
require 'sys'
require 'torch'
require 'cutorch'
require 'dataset-mnist'
require 'optim'
dofile '../cifar.torch/provider.lua'
dofile 'distilling_criterion.lua'

local cnnGameEnv = torch.class('cnnGameEnv')

local function cast(t)
    return t:cuda()
end

function cnnGameEnv:__init(opt)
	print('option from cmd: \n', opt)
	self.base = '../save/'
	function isExist(name)
		local f = io.open(name, "r")
		if f ~= nil then io.close(f) return true else return false end
	end
	print('init game environment.\n')
	torch.manualSeed(os.time())
	torch.setdefaulttensortype('torch.FloatTensor')
    self.dataset = opt.dataset
    self.model = self:create_mlp_model()
    self.model:cuda()
    self.parameters, self.gradParameters = self.model:getParameters()
    self.criterion = nn.CrossEntropyCriterion():cuda()
    self.trsize = 60000
    self.tesize = 10000
    local geometry = {32, 32}
    if self.dataset == 'MNIST' then
        self.trainData = mnist.loadTrainSet(self.trsize, geometry)
        self.trainData:normalizeGlobal(mean, std)
        self.testData = mnist.loadTestSet(self.tesize, geometry)
        self.testData:normalizeGlobal(mean, std)
    else
        if not isExist('base' .. 'provider.t7') then
            -- TODO: add provider loading code
        end
        local provider = torch.load('../save/provider.t7')
        provider.trainData.data = provider.trainData.data:float()
        provider.testData.data = provider.testData.data:float()
        self.trainData = provider.trainData
        self.testData = provider.testData
    end
    self.classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
    self.confusion = optim.ConfusionMatrix(self.classes)
    self.batchsize = opt.batchsize
    self.total_batch_number = math.ceil(self.trsize / self.batchsize) - 1 --mini-batch number in one epoch
    self.learningRate = opt.learningRate  --init learning rate
    self.weightDecay = 0
    self.momentum = 0
    self.finalerr = 0.0001
    self.epoch = 0
    self.episode = 0
    self.batchindex = 1  --train batch by batch
    self.channel = 1
    self.err = 2.5  --start error
    self.mapping = {}
	self.terminal = false  --whether a episode game stop
    self.datapointer = 1  --serve as a pointer, scan all the training data in one iteration.
	self.max_epoch = opt.max_epoch   -- # of epoch in one episode
    if opt.DQN_off == 1 then
        self.DQN_off = true
    else
        if opt.extra_loss == 1 then
            self.extra_loss = true
            --print("extra_loss on".. self.extra_loss .. "\n\n\n\n\n")
            self.w1 = torch.load(self.base..'cnnfilter1.t7')
            self.w2 = torch.load(self.base..'cnnfilter2.t7')
            self.w3 = torch.load(self.base..'cnnfilter3.t7')
            self.w4 = torch.load(self.base..'cnnfilter4.t7')
        end
    end
	if opt.distilling_on == 1 then -- if knowledge distilling is on
        print("distilling configuring here..")
        self.distilling_start_epoch = 0
        self.distilling_on = true
        self.softDataset = self.trainData
        self.temp = opt.temp -- distilling temperature
        self.sm = nn.SoftMax():cuda()
        self.distilling_loss = opt.distilling_loss
        self.soft_criterion = DistillingCriterion(0.1, 3, self.distilling_loss)-- DistillingCriterion(0.1, opt.temp, 'L2')
        -- read from previously saved log or not
        self.history = true -- to load a soft_label from *.t7 or not
    end

end

function cnnGameEnv:create_mlp_model()
    local model = nn.Sequential()
    if self.dataset == 'MNIST' then
        print('Building a MNIST model.\n')
        -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
        model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
        model:add(nn.Tanh())
        model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
        -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
        model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
        model:add(nn.Tanh())
        model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        -- stage 3 : standard 2-layer MLP:
        model:add(nn.Reshape(64*2*2))
        model:add(nn.Linear(64*2*2, 200))
        model:add(nn.Tanh())
        model:add(nn.Linear(200, 10))
    else
        print('Building a CIFAR model.\n')
        model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
        model:add(cast(dofile('models/vgg_bn_drop.lua')))
        model:get(2).updateGradInput = function(input) return end
    end
    -- model:add(nn.LogSoftMax())
    return model
end

cnt_not_match = 0

function cnnGameEnv:train()
	-- The training process is modified from
    -- [https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua]
    local dataset = self.trainData
    -- epoch tracker
    self.batchindex = self.batchindex or 1
    local trainError = 0
    -- do one mini-batch
	local bsize = math.min(self.batchsize, dataset:size()-self.datapointer + 1)
    print("batch zise of this iteration:", bsize)
    local inputs = torch.CudaTensor(bsize, self.channel, 32, 32)
    local targets = torch.CudaTensor(bsize)
    for i = 1, bsize do
        local idx = self.datapointer
        local sample = dataset[idx]
        local input = sample[1]:clone()
        local _, target = sample[2]:clone():max(1)
        -- print('_: ' , _)
        target = target:squeeze()
        -- after squeeze, it become a scalar label
        inputs[i] = input
        targets[i] = target
        self.datapointer = self.datapointer + 1
    end

    if self.distilling_on and self.distilling_start == 1 then
        if self.soft_label == nil then
            self.soft_label = torch.load(self.base .. 'soft_label.t7')
            print("load soft_label", self.soft_label[1])
            for i = 1, 938 do
                self.soft_label[i] = self.soft_label[i]:cuda()
            end
        end
        print(self.batchindex, self.datapointer)
        assert(self.batchindex * self.batchsize == self.datapointer - 1)
        targets = { soft_target = self.soft_label[self.batchindex], labels = targets }


        -- test softlabel here! TODO: move to a stand alone test part
        local _, idx = torch.max(targets.soft_target, 2)
        for i = 1, bsize do
            if idx[i]:squeeze() ~= targets.labels[i] then
                cnt_not_match = cnt_not_match + 1
                print(idx[i])
                --print(targets.soft_target[i])
                --print(targets.labels[i])
            end
        end
        print("not match!!!!!!!!!!!!!!!!!", cnt_not_match)
        -- DistillingCriterion targets format: {soft_target, labels}
    end

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        collectgarbage()
        if x ~= self.parameters then
            self.parameters:copy(x)
        end
        self.gradParameters:zero()
        local err, df_do
        local output = self.model:forward(inputs)
        print("self.distilling: ", self.distilling_on, self.distilling_start)
        if self.distilling_on and self.distilling_start == 1 then
            print("using soft criterion. ")
            err = self.soft_criterion:forward(output, targets)
            df_do = self.soft_criterion:backward(output, targets)
        else
            print("using normal criterion. ")
            err = self.criterion:forward(output, targets)
            df_do = self.criterion:backward(output, targets)
        end
        self.model:backward(inputs, df_do)
        for i = 1, bsize do
            if self.distilling_on and self.distilling_start ~= nil then
                self.confusion:add(output[i], targets.labels[i])
            else
                self.confusion:add(output[i], targets[i])
            end
        end
        self.err = err
        return f, self.gradParameters
    end
    -- optimize on current mini-batch
    self.config = self.config or {learningRate = self.learningRate,
                  momentum = self.momentum,
                  learningRateDecay = 5e-7}
    optim.sgd(feval, self.parameters, self.config)
    print (self.confusion)
    local trainAccuracy = self.confusion.totalValid * 100 
    print(trainAccuracy)
    self.confusion:zero()
    return trainAccuracy, trainError
end

function cnnGameEnv:getDistillingLabel()
    print("Time to get soft label now!")
    local dataset = self.trainData
    -- epoch tracker
    self.batchindex = self.batchindex or 1
    -- do one mini-batch
    local getMiniBatchLabel = function(x)
        print("getMiniBatchLabel!")
        local bsize = math.min(self.batchsize, dataset:size()-self.datapointer + 1)
        local inputs = torch.CudaTensor(bsize, self.channel, 32, 32)
        -- local targets = torch.CudaTensor(bsize)
        for i = 1, bsize do
            local idx = self.datapointer
            local sample = dataset[idx]
            local input = sample[1]:clone()
            -- local _, target = sample[2]:clone():max(1)
            -- target = target:squeeze()
            inputs[i] = input
            -- targets[i] = target
            self.datapointer = self.datapointer + 1
        end
        -- getsoftLabel
        local output = self.model:forward(inputs)
        --print("output here: ", output)
        --print("twt: ", twt)
        -- print("distilling temperature: ", self.temp)
        output = self.sm:forward(output / self.temp):clone()-- temperature
        --print("return output")
        return output
    end
    local soft_label = {}

    self.datapointer = 1
    for i = 1, self.total_batch_number do
        soft_label[i] = getMiniBatchLabel():clone()
    end
    self.soft_label = soft_label
    self.datapointer = 1 -- reset datapointer

    torch.save( self.base .. 'soft_label.t7', soft_label)

    sys.sleep(10)
    return soft_label
end

function cnnGameEnv:test()
	--The testing process is also modified from
    -- [https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua]
    local dataset = self.testData
    local testError = 0
    print('<trainer> on testing Set:')
    for t = 1, self.tesize do
       -- disp progress
       xlua.progress(t, self.tesize)
       -- get new sample
       local input = torch.CudaTensor(1, self.channel, 32, 32)
       input[1] = dataset.data[t]
       local target = dataset.labels[t]
       -- test sample, pred is the probability assuption over ten class?
       local pred = self.model:forward(input[1])
	   -- print("I want to see what is in pred:")
	   -- print(pred:view(10))
	   -- print(target)
       self.confusion:add(pred:view(10), target)
       -- compute error
       local err = self.criterion:forward(pred, target)
       testError = testError + err
    end
    if self.verbose then
        print(self.confusion)
    end
    local testAccuracy = self.confusion.totalValid * 100
    self.confusion:zero()
    return testAccuracy, testError
end

function cnnGameEnv:regression(targets, weights, layernum)
    local input_neural_number, output_neural_number
	if layernum == 1 then 
		input_neural_number = 32
		output_neural_number = 25
	elseif layernum == 2 then
		input_neural_number = 64
		output_neural_number = 800
	elseif layernum == 3 then
		input_neural_number = 200
		output_neural_number = 256
	elseif layernum == 4 then
		input_neural_number = 10
		output_neural_number = 200
	end

	--input_neural_number = 32
	--output_neural_number = 25
    --local targets = self.filter:cuda()
    --local weights = self.model:get(4).weight:cuda()
	targets = targets:cuda()
	weights = weights:cuda()
--    local regressweights = torch.CudaTensor(10, 256)
    local reg_data = torch.CudaTensor(input_neural_number, output_neural_number):fill(1)
    -- https://github.com/torch/nn/blob/master/doc/simple.md#nn.CMul
    local reg_model = nn.Sequential()
    reg_model:add(nn.CMul(output_neural_number))
	reg_model:cuda()
    -- how to set weights into reg_model? set in line 156
    function regGradUpdate(reg_model, reg_x, reg_y, reg_criterion, regLearningRate)
        local reg_pred = reg_model:forward(reg_x)
        local reg_err = reg_criterion:forward(reg_pred, reg_y)
        local regGradCriterion = reg_criterion:backward(reg_pred, reg_y)
        reg_model:zeroGradParameters()
        reg_model:backward(reg_x, regGradCriterion)
        reg_model:updateParameters(regLearningRate)
        return reg_err
    end

    for i = 1, input_neural_number do
		reg_model:get(1).weight:copy(weights[i])
        -- do 3 iterations of regression
        for j = 1, 3 do
            local err = regGradUpdate(reg_model, reg_data[i], targets[i], nn.MSECriterion():cuda(), 0.01)
        end
		weights[i]:copy(reg_model:get(1).weight)
    end
    -- need to set weights back here: set back in line 161
end

function cnnGameEnv:reward(verbose, filter, tstate)
	verbose = verbose or false
    local reward = 0
    if self.err then
        reward = 1 / math.abs(self.err - self.finalerr) 
    end
	if (verbose) then
        print ('finalerr: ' .. self.finalerr)
		if self.err then print ('err: ' .. self.err) end
		print ('reward: '.. reward)
	end
	print ('err: ' .. self.err)
	print ('reward is: ' .. reward)
	return reward
end

function cnnGameEnv:getActions()
	local gameActions = {}
	for i=1, 3 do
		gameActions[i] = i
	end
	return gameActions
end

function cnnGameEnv:getState(verbose) --state is set in cnn.lua
	local verbose = verbose or false
	--return state, reward, terminal
	local tstate = self.model:get(1).weight 
	local size = tstate:size()[1] * tstate:size()[2]  
	print(size)
    local filter = self.filter
	local reward = self:reward(verbose, filter, tstate)
    return tstate, reward, self.terminal
end

function cnnGameEnv:step(action, tof)
	print('step')
	io.flush()
	--[[
		action 1: increase
		action 2: decrease
		action 3: unchanged	ca
	]]
	local delta = 0.005
	local minlr = 0.005
	local maxlr = 1.0

    if not self.DQN_off then
        if (action == 1) then
            self.learningRate = math.min(self.learningRate + delta, maxlr);
        elseif (action == 2) then
            self.learningRate = math.max(self.learningRate - delta, minlr);
        end
    end

    print('<trainer> on training set:' .. 'epoch #' .. self.epoch .. ', batchindex ' .. self.batchindex)
    local trainAcc, trainErr = self:train()
    self.trainAcc = self.trainAcc or 0
    self.trainAcc = self.trainAcc + trainAcc
	print('batchindex = '.. self.batchindex)
	print('totalbatchindex = '.. self.total_batch_number)

    --print(self.extra_loss)
    --sys.sleep(10)
    if self.extra_loss and self.epoch % self.max_epoch <= 5 then   --let cnn train freely after 5 epoches.
    --print("it works!")
    --sys.sleep(5)
		local w1 = self.model:get(1).weight
		local w2 = self.model:get(4).weight
		local w3 = self.model:get(8).weight
		local w4 = self.model:get(10).weight
		self:regression(self.w1, w1, 1)
		self:regression(self.w2, w2, 2)
		self:regression(self.w3, w3, 3)
		self:regression(self.w4, w4, 4)
    end

--    -- test get_label
--    if self.datapointer > 5 * self.batchsize then
--        local res = self:getDistillingLabel()
--        print("result of distilling: ", res)
--    end

    if self.distilling_on and self.history and self.distilling_start == nil then
        self.distilling_start = 1
        self.soft_label = torch.load(self.base .. 'soft_label.t7')
        print("Load soft_label success!")
        sys.sleep(5)
    end

    if self.batchindex == self.total_batch_number then
        self.datapointer = 1 --reset the pointer
        self.trainAcc = self.trainAcc / self.total_batch_number
        print ('trainAcc = ' .. self.trainAcc)
--        if self.epoch % self.max_epoch == 0 then
--            self.episode = self.episode + 1
--        end
        local outputtrain = self.dataset .. 'train_lr_KL_0.1_dqnon_' .. self.episode .. '.log'--'basetrain.log'--'baseline_raw_train.log'
        local outputtest = self.dataset .. 'test_lr_KL_0.1_dqnon_' .. self.episode .. '.log'--'basetest.log'--'baseline_raw_test.log'
        os.execute('echo ' .. self.trainAcc .. ' >> logs/' .. outputtrain)
        self.trainAcc = 0
        local testAcc,  testErr = self:test()
        print("testacc: "..testAcc.."\n\n")
        os.execute('echo ' .. testAcc .. ' >> logs/' .. outputtest)
        self.batchindex = 0 --reset the batch pointer
        self.epoch = self.epoch + 1
        print('epoch = ' .. self.epoch)
        sys.sleep(4)

        if self.epoch > 0 and self.epoch % self.max_epoch == 0 then
            -- start distilling
            if self.distilling_start == nil then
                self.distilling_start = 1
                self:getDistillingLabel()
            end
            --reset learning rate
			self.learningRate = 0.05
			self.terminal = true
            --reset model
            self.model = self:create_mlp_model()
            self.model:cuda()
            self.parameters, self.gradParameters = self.model:getParameters()
        end
    end
    self.batchindex = self.batchindex + 1
	return self:getState()
end

function cnnGameEnv:nextRandomGame()

end

function cnnGameEnv:newGame()

end

function cnnGameEnv:nObsFeature()

end

