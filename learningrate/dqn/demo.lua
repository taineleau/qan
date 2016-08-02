require 'cnnGameEnv'

local cnn = cnnGameEnv()
for s = 1,6000 do
	cnn:step()
end
--cnn:loaddata()
