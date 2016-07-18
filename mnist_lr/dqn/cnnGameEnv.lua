require 'cunn'
require 'sys'
require 'torch'
require 'cutorch'
require 'dataset-mnist'
require 'optim'
dofile 'distilling_criterion.lua'

local cnnGameEnv = torch.class('cnnGameEnv')

function cnnGameEnv:__init(opt)
	print(opt)
	self.base = '../save/'
--	function check(name)
--		local f = io.open(name, "r")
--		if f ~= nil then io.close(f) return true else return false end
--	end
	print('init game environment')
	torch.manualSeed(os.time())
	torch.setdefaulttensortype('torch.FloatTensor')
    self.model = self:create_mlp_model()
    self.model:cuda()
    self.parameters, self.gradParameters = self.model:getParameters()
    self.criterion = nn.CrossEntropyCriterion():cuda()
    -- self.criterion = nn.ClassNLLCriterion():cuda()
    self.trsize = 60000
    self.tesize = 10000
    local geometry = {32, 32}
    self.trainData = mnist.loadTrainSet(self.trsize, geometry)
    self.trainData:normalizeGlobal(mean, std)
    self.testData = mnist.loadTestSet(self.tesize, geometry)
    self.testData:normalizeGlobal(mean, std)
    self.classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
    self.confusion = optim.ConfusionMatrix(self.classes)
    self.batchsize = 64
    self.total_batch_number = math.ceil(self.trsize / self.batchsize)  --mini-batch number in one epoch
    self.learningRate = 0.05  --init learning rate
    self.weightDecay = 0
    self.momentum = 0
	--load filter for regression
	self.w1 = torch.load(self.base..'cnnfilter1.t7')
	self.w2 = torch.load(self.base..'cnnfilter2.t7')
	self.w3 = torch.load(self.base..'cnnfilter3.t7')
	self.w4 = torch.load(self.base..'cnnfilter4.t7')
    self.finalerr = 0.0001 
    self.epoch = 0
    self.batchindex = 1  --train batch by batch
    self.channel = 1
    self.err = 2.5  --start error
    self.mapping = {}
	self.terminal = false  --whether a episode game stop
    self.datapointer = 1  --serve as a pointer, scan all the training data in one iteration.
	self.max_epoch = 20
	if opt.distilling_on == 1 then
        print("distilling configuring here..")
        self.distilling_start_epoch = 0
        self.distilling_on = true
        self.softDataset = self.trainData
        self.temp = opt.temp -- distilling temperature
        self.sm = nn.SoftMax():cuda()
        self.soft_criterion = DistillingCriterion(0.9, opt.temp, 'L2')
        --DistillingCriterion.__init(0.9, opt.temp, 'L2')
    end

end

function cnnGameEnv:create_mlp_model()
    local model = nn.Sequential()
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
    -- model:add(nn.LogSoftMax())
    return model
end

function cnnGameEnv:train()
	-- The training process is modified from
    -- [https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua]
    local dataset = self.trainData
    -- epoch tracker
    self.batchindex = self.batchindex or 1
    local trainError = 0
    -- do one mini-batch
	local bsize = math.min(self.batchsize, dataset:size()-self.datapointer + 1)
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

    if self.distilling_on and self.epoch >= self.distilling_start_epoch then
        if self.soft_label == nil then
            self.soft_label = torch.load(self.base .. 'soft_label.t7')
            print("load soft_label", self.soft_label[1])
            for i = 1, 938 do
                self.soft_label[i] = self.soft_label[i]:cuda()
            end
        end
        targets = { soft_target = self.soft_label[self.batchindex], labels = targets}
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
        -- print("linear output: ", output)
        if self.distilling_on and self.epoch >= self.distilling_start_epoch then
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
            self.confusion:add(output[i], targets.labels[i])
        end
        self.err = err
        return f, self.gradParameters
    end
    -- optimize on current mini-batch
    local config = config or {learningRate = self.learningRate,
                  momentum = self.momentum,
                  learningRateDecay = 5e-7}
    optim.sgd(feval, self.parameters, config)
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

    sys.sleep(100)
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
    -- print confusion matrix
    print(self.confusion)
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
	for i=1,3 do
		gameActions[i] = i
	end
	return gameActions
end

function cnnGameEnv:getState(verbose) --state is set in cnn.lua
	verbose = verbose or false
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
		action 3: unchanged	
	]]
	local delta = 0.005
	local minlr = 0.005
	local maxlr = 1.0
	local outputtrain = 'train_lr_baseline1.log'--'basetrain.log'--'baseline_raw_train.log'
	local outputtest = 'test_lr_baseline1.log'--'basetest.log'--'baseline_raw_test.log'

    --if (action == 1) then 
    --    self.learningRate = math.min(self.learningRate + delta, maxlr);
    --elseif (action == 2) then 
    --    self.learningRate = math.max(self.learningRate - delta, minlr);
    --end

    --sys.sleep(10)
 --   torch.save('../save/soft_labelago.t7', delta)

    print('<trainer> on training set:' .. 'epoch #' .. self.epoch .. ', batchindex ' .. self.batchindex)
    local trainAcc, trainErr = self:train()
    self.trainAcc = self.trainAcc or 0
    self.trainAcc = self.trainAcc + trainAcc
	print('batchindex = '.. self.batchindex)
	print('totalbatchindex = '.. self.total_batch_number)

    if self.epoch % self.max_epoch <= 5 then   --let cnn train freely after 5 epoches.
		--local w1 = self.model:get(1).weight
		--local w2 = self.model:get(4).weight
		--local w3 = self.model:get(8).weight
		--local w4 = self.model:get(10).weight
		--self:regression(self.w1, w1, 1)
		--self:regression(self.w2, w2, 2)
		--self:regression(self.w3, w3, 3)
		--self:regression(self.w4, w4, 4)
    end
--    -- test get_label
--    if self.datapointer > 5 * self.batchsize then
--        local res = self:getDistillingLabel()
--        print("result of distilling: ", res)
--    end

    if self.batchindex == self.total_batch_number then
        self.datapointer = 1 --reset the pointer
        self.trainAcc = self.trainAcc / self.total_batch_number
        print ('trainAcc = ' .. self.trainAcc)
        os.execute('echo ' .. self.trainAcc .. ' >> logs/' .. outputtrain)
        self.trainAcc = 0
        local testAcc,  testErr = self:test()
        os.execute('echo ' .. testAcc .. ' >> logs/' .. outputtest)
        self.batchindex = 1 --reset the batch pointer
        self.epoch = self.epoch + 1
        print('epoch = ' .. self.epoch)
        sys.sleep(10)

        if self.epoch == 0 and self.distilling_on then
            self.soft_label = torch.load(self.base .. 'soft_label.t7')
            print("Load soft_label success!")
            sys.sleep(1)
        end
--        if self.epoch == 1 and self.distilling_on then
--            -- TODO: add soft target
--            self:getDistillingLabel()
--        end

        if self.epoch > 0 and self.epoch % self.max_epoch == 0 then
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

