--[[ Custom criterion for Dark Knowledge Transfer

cf Hinton et al. - Distilling the Knowledge in a Neural Network 

this code is modified from: https://github.com/natoromano/specialistnets

]]--

local DistillingCriterion, parent = torch.class('DistillingCriterion',
                                                   'nn.Criterion')

function DistillingCriterion:__init(alpha, temp, soft_loss, verbose)
    -- alpha: soft cross-entropy weight
    -- temp: temperature
    -- soft_loss: 'KL', 'MSE' or 'L1'
    parent.__init(self)
    self.temp = temp or 1.0
    self.alpha = alpha or 0.9
    self.soft_loss = soft_loss or 'KL'
    self.supervised = (self.alpha < 1.0)
    self.verbose = verbose or false
    self.sm = nn.SoftMax():cuda()
    self.lsm = nn.LogSoftMax():cuda()
    self.ce_crit = nn.CrossEntropyCriterion():cuda()
    if self.soft_loss == 'KL' then
        self.kl_crit = nn.DistKLDivCriterion():cuda()
    elseif self.soft_loss == 'L2' then
        print("choose L2.")
        self.mse_crit = nn.MSECriterion():cuda()
        self.mse_crit.sizeAverage = false
    else
        error('Invalid input as soft_loss')
    end
end

function DistillingCriterion:updateOutput(input, target)
    -- input: raw scores from the model
    -- target.labels = ground truth labels
    -- target.scores = raw scores from the master
    -- local soft_target = self.sm:forward(target.scores / self.temp):clone()
    local soft_target = target.soft_target
    print("input", input)
--    print("soft_target", soft_target)
--    if soft_target.size()[1] != input.size()[1] then
--        soft_target.reshape(soft_target, input.size()[1], soft_target.size()[2])
--    end
    --print("labels:\n", target.labels)
    if self.soft_loss == 'KL' then
      local log_probs = self.lsm:forward(input / self.temp)
      if self.supervised then
          self.output = self.ce_crit:forward(input, 
                                    target.labels) * (1 - self.alpha)
          if self.verbose then
            local str = string.format('CE/KL loss: %1.0e/%1.0e', self.output,
                    self.kl_crit:forward(log_probs, soft_target) * self.alpha)
            print(str)
          end 
          self.output = self.output +
              self.kl_crit:forward(log_probs, soft_target) * self.alpha
      else
          self.output = self.kl_crit:forward(log_probs, soft_target)
      end
    else
      local probs = self.sm:forward(input / self.temp)
      if self.supervised then
          print("soft~~~~", target.labels)
          self.output = self.ce_crit:forward(input, target.labels)
          self.output = self.output * (1 - self.alpha) + 
              self.mse_crit:forward(probs, soft_target) * self.alpha
      else
          self.output = self.mse_crit:forward(probs, soft_target)
      end
    end
    return self.output
end

function DistillingCriterion:updateGradInput(input, target)
    self.mask = target.labels:eq(0)
    -- local soft_target = self.sm:forward(target.scores:div(self.temp)):clone()
    local soft_target = target.soft_target
    if self.soft_loss == 'KL' then
      local log_probs = self.lsm:forward(input / self.temp)
      if self.supervised then
          local grad_ce = self.ce_crit:backward(input, 
                                      target.labels) * (1 - self.alpha)
          local grad_kl = self.kl_crit:backward(log_probs, 
                                      soft_target) * (self.alpha)
          grad_kl = self.lsm:backward(input:div(self.temp),grad_kl) * self.temp
          --grad_kl is multiplied by T^2 as recommended by Hinton et al. 
          self.gradInput = grad_ce + grad_kl
          if self.verbose then
            local str = string.format('CE/KL grad: %1.0e/%1.0e', 
                                      grad_ce:norm(), grad_kl:norm())
            print(str)
          end
      else
          local grad_kl = self.kl_crit:backward(log_probs, soft_target)
          grad_kl = self.lsm:backward(input:div(self.temp),grad_kl) * self.temp
          -- grad_kl is multiplied by T^2 as recommended by Hinton et al. 
          self.gradInput = grad_kl        
      end
    else
      local probs = self.sm:forward(input:div(self.temp))
      if self.supervised then
          local grad_ce = self.ce_crit:backward(input, 
                                      target.labels) * (1 - self.alpha)
          local grad_mse = self.mse_crit:backward(probs, 
                                      soft_target) * (self.alpha)
          grad_mse = self.sm:backward(input:div(self.temp),grad_mse) * self.temp
          -- grad_kl is multiplied by T^2 as recommended by Hinton et al. 
          self.gradInput = grad_ce + grad_mse
      else
          local grad_mse = self.mse_crit:backward(probs, soft_target)
          grad_mse = self.sm:backward(input:div(self.temp),grad_mse) * self.temp
          -- grad_kl is multiplied by T^2 as recommended by Hinton et al. 
          self.gradInput = grad_mse        
      end
    end
    return self.gradInput
end
