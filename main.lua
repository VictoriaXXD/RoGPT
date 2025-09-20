local GPT = {}
GPT.__index = GPT

--Libraries for different lua Versions
local math_sqrt = math.sqrt
local math_exp = math.exp
local math_log = math.log
local math_max = math.max
local math_min = math.min
local math_random = math.random

--Helper Functions for vectors and matrices
local function zeros(n)
	local t = {}
	for i = 1, n do t[i] = 0 end
	return t
end

local function zerosMatrix(rows, cols)
	local m = {}
	for i = 1, rows do
		m[i] = zeros(cols)
	end
	return m
end

function matAdd(a, b)
	local rows = #a
	local cols = #b[1]
	local c = {}
	for i = 1, rows do
		c[i] = {}
		for j = 1, cols do
			c[i][j] = a[i][j] + b[i][j]
		end
	end
	return c
end

local function randomMatrix(rows, cols, scale)
	scale = scale or 0.1
	local m = {}
	for i = 1, rows do
		m[i] = {}
		for j = 1, cols do
			m[i][j] = (math_random() - 0.5) * 2 * scale
		end
	end
	return m
end

local function vecCopy(v)
	local out = {}
	for i = 1, #v do out[i] = v[i] end
	return out
end

local function vecAdd(a, b)
	local out = {}
	for i = 1, #a do out[i] = a[i] + b[i] end
	return out
end

local function vecAddInPlace(a, b)
	for i = 1, #a do a[i] = a[i] + b[i] end
end

local function vecSub(a, b)
	local out = {}
	for i = 1, #a do out[i] = a[i] - b[i] end
	return out
end

local function vecMulScalar(v, s)
	local out = {}
	for i = 1, #v do out[i] = v[i] * s end
	return out
end

local function vecHadamard(a, b)
	local out = {}
	for i = 1, #a do out[i] = a[i] * b[i] end
	return out
end

local function matVecMul(mat, vec)
	local out = {}
	for i = 1, #mat do
		local s = 0
		for j = 1, #vec do
			s = s + mat[i][j] * vec[j]
		end
		out[i] = s
	end
	return out
end

local function matVecMulTranspose(mat, vec)
	local cols = #mat[1]
	local out = zeros(cols)
	for j = 1, cols do
		local s = 0
		for i = 1, #mat do
			s = s + mat[i][j] * vec[i]
		end
		out[j] = s
	end
	return out
end

local function outerProduct(vecA, vecB)
	local m = {}
	for i = 1, #vecA do
		m[i] = {}
		for j = 1, #vecB do
			m[i][j] = vecA[i] * vecB[j]
		end
	end
	return m
end

local function zerosMatrixGrad(rows, cols)
	local m = {}
	for i = 1, rows do
		m[i] = {}
		for j = 1, cols do m[i][j] = 0 end
	end
	return m
end

local function zerosVectorGrad(n)
	return zeros(n)
end

local function addMatrixGrad(gradMat, addMat)
	for i = 1, #gradMat do
		for j = 1, #gradMat[i] do
			gradMat[i][j] = gradMat[i][j] + addMat[i][j]
		end
	end
end

local function addVectorGrad(gradVec, addVec)
	for i = 1, #gradVec do
		gradVec[i] = gradVec[i] + addVec[i]
	end
end

--Activation functions
local function relu(x)
	return x > 0 and x or 0
end

local function drelu(x)
	return x > 0 and 1 or 0
end

local function softmax(x)
	local maxVal = -math.huge
	for _, v in ipairs(x) do
		if v > maxVal then maxVal = v end
	end
	local exps = {}
	local sum = 0
	for i = 1, #x do
		exps[i] = math_exp(x[i] - maxVal)
		sum = sum + exps[i]
	end
	for i = 1, #exps do
		exps[i] = exps[i] / sum
	end
	return exps
end

GPT.softmax = softmax

--LayerNorm (without learnable parameters)
local function layerNorm(x)
	local n = #x
	local mean = 0
	for i = 1, n do mean = mean + x[i] end
	mean = mean / n
	local variance = 0
	for i = 1, n do
		variance = variance + (x[i] - mean)^2
	end
	variance = variance / n
	local std = math_sqrt(variance + 1e-5)
	local out = {}
	for i = 1, n do
		out[i] = (x[i] - mean) / std
	end
	return out, mean, std
end

--LayerNorm Backward
local function layerNormBackward(dout, x, mean, std)
	local n = #x
	local dx = {}
	local dmean = 0
	local dvar = 0
	for i = 1, n do
		dx[i] = dout[i] / std
		dmean = dmean - dout[i] / std
		dvar = dvar - 0.5 * dout[i] * (x[i] - mean) / (std^3)
	end
	for i = 1, n do
		dx[i] = dx[i] + dmean / n + 2 * dvar * (x[i] - mean) / n
	end
	return dx
end

--Causal Mask (for Attention Scores)
local function causalMask(seqLen)
	local mask = {}
	for i = 1, seqLen do
		mask[i] = {}
		for j = 1, seqLen do
			mask[i][j] = (j <= i) and 0 or -math.huge
		end
	end
	return mask
end

--Adam Optimizer helper functions
local function adamInitMatrix(rows, cols)
	return {m = zerosMatrix(rows, cols), v = zerosMatrix(rows, cols), t = 0}
end

local function adamInitVector(n)
	return {m = zeros(n), v = zeros(n), t = 0}
end

local function adamUpdateParam(param, grad, state, lr, beta1, beta2, eps)
	state.t = state.t + 1
	local m = state.m
	local v = state.v
	local t = state.t
	local b1t = 1 - beta1^t
	local b2t = 1 - beta2^t
	if type(param[1]) == "table" then
		for i = 1, #param do
			for j = 1, #param[i] do
				m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad[i][j]
				v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad[i][j] * grad[i][j]
				local m_hat = m[i][j] / b1t
				local v_hat = v[i][j] / b2t
				param[i][j] = param[i][j] - lr * m_hat / (math_sqrt(v_hat) + eps)
			end
		end
	else
		for i = 1, #param do
			m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
			v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]
			local m_hat = m[i] / b1t
			local v_hat = v[i] / b2t
			param[i] = param[i] - lr * m_hat / (math_sqrt(v_hat) + eps)
		end
	end
end

--Slice Vector
local function slice(vec, startIdx, endIdx)
	local out = {}
	for i = startIdx, endIdx do
		table.insert(out, vec[i])
	end
	return out
end

--Split vector in heads (for multihead attention)
local function splitHeads(x, heads, dHead)
	local out = {}
	for h = 1, heads do
		out[h] = slice(x, (h - 1) * dHead + 1, h * dHead)
	end
	return out
end

--Combining the heads
local function combineHeads(headsTable)
	local out = {}
	for h = 1, #headsTable do
		for i = 1, #headsTable[h] do
			table.insert(out, headsTable[h][i])
		end
	end
	return out
end

--CrossEntropy Loss Gradients (Logits and Target Index)
local function crossEntropyGrad(logits, targetIdx)
	local probs = softmax(logits)
	local grad = {}
	for i = 1, #probs do grad[i] = probs[i] end
	grad[targetIdx] = grad[targetIdx] - 1
	return grad
end

--Sum for LayerNorm Grads
function sumVecMul(dout, x, mean, invStd)
	local s = 0
	for i = 1, #x do
		s = s + dout[i] * (x[i] - mean) * invStd
	end
	return s
end

--FNN feedforward layer with ReLU and backward
local function feedForwardForward(x, W1, b1, W2, b2)
	local h1 = {}
	for i = 1, #W1 do
		local s = 0
		for j = 1, #x do
			s = s + W1[i][j] * x[j]
		end
		s = s + b1[i]
		h1[i] = relu(s)
	end
	local out = {}
	for i = 1, #W2 do
		local s = 0
		for j = 1, #W2[1] do
			s = s + W2[i][j] * h1[j]
		end
		s = s + b2[i]
		out[i] = s
	end
	return out, h1
end

local function feedForwardBackward(dout, h1, x, W2, W1)
	local dW2 = zerosMatrix(#W2, #W2[1])
	local db2 = zeros(#W2)
	local dh1 = zeros(#h1)
	for i = 1, #W2 do
		db2[i] = dout[i]
		for j = 1, #W2[1] do
			dW2[i][j] = dout[i] * h1[j]
			dh1[j] = dh1[j] + dout[i] * W2[i][j]
		end
	end

	local dW1 = zerosMatrix(#W1, #W1[1])
	local db1 = zeros(#W1)
	local dx = zeros(#x)
	for i = 1, #W1 do
		local gradRelu = drelu(h1[i])
		local grad = dh1[i] * gradRelu
		db1[i] = grad
		for j = 1, #W1[1] do
			dW1[i][j] = grad * x[j]
			dx[j] = dx[j] + grad * W1[i][j]
		end
	end

	return dx, dW1, db1, dW2, db2
end

--Multihead Attention Forward
local function multiHeadAttentionForward(xSeq, layer, heads, dModel, mask)
	--xSeq: seqLen x dModel
	local seqLen = #xSeq
	local dHead = dModel // heads

	--Prepare all Q,K,V
	local Qs = {}
	local Ks = {}
	local Vs = {}
	for t = 1, seqLen do
		Qs[t] = matVecMul(layer.qW, xSeq[t])
		Ks[t] = matVecMul(layer.kW, xSeq[t])
		Vs[t] = matVecMul(layer.vW, xSeq[t])
	end

	--Split Heads
	local QsH = {}
	local KsH = {}
	local VsH = {}
	for t = 1, seqLen do
		QsH[t] = splitHeads(Qs[t], heads, dHead)
		KsH[t] = splitHeads(Ks[t], heads, dHead)
		VsH[t] = splitHeads(Vs[t], heads, dHead)
	end

	--Attention Scores + Probs
	local attnScores = {}
	local attnProbs = {}
	for t = 1, seqLen do
		attnScores[t] = {}
		attnProbs[t] = {}
		for h = 1, heads do
			local scores = {}
			for j = 1, seqLen do
				if mask[t][j] == 0 then
					local s = 0
					for d = 1, dHead do
						s = s + QsH[t][h][d] * KsH[j][h][d]
					end
					s = s / math_sqrt(dHead)
					scores[j] = s
				else
					scores[j] = -math.huge
				end
			end
			attnScores[t][h] = scores
			attnProbs[t][h] = softmax(scores)
		end
	end

	--Attention Output: weighted sum over Vs
	local headsOut = {}
	for t = 1, seqLen do
		headsOut[t] = {}
		for h = 1, heads do
			local ctx = zeros(dHead)
			for j = 1, seqLen do
				for d = 1, dHead do
					ctx[d] = ctx[d] + VsH[j][h][d] * attnProbs[t][h][j]
				end
			end
			headsOut[t][h] = ctx
		end
	end

	--Concat Heads and Out projection
	local outSeq = {}
	for t = 1, seqLen do
		local concat = combineHeads(headsOut[t])
		outSeq[t] = vecAdd(matVecMul(layer.outW, concat), layer.outB)
	end

	--Cache for backprop
	local cache = {
		xSeq = xSeq,
		Qs = Qs,
		Ks = Ks,
		Vs = Vs,
		QsH = QsH,
		KsH = KsH,
		VsH = VsH,
		attnScores = attnScores,
		attnProbs = attnProbs,
		headsOut = headsOut,
		outSeq = outSeq,
	}
	return outSeq, cache
end

--Multihead Attention Backward
local function multiHeadAttentionBackward(doutSeq, cache, layer, heads, dModel, mask)
	local seqLen = #cache.xSeq
	local dHead = dModel // heads

	--Init grads
	local dQ = {}
	local dK = {}
	local dV = {}
	for t = 1, seqLen do
		dQ[t] = zeros(dModel)
		dK[t] = zeros(dModel)
		dV[t] = zeros(dModel)
	end

	local dOutW = zerosMatrix(#layer.outW, #layer.outW[1])
	local dOutB = zeros(#layer.outB)
	local dConcatHeads = {}

	--Backprop Out projection
	for t = 1, seqLen do
		--doutSeq[t] (dim dModel)
		local concat = combineHeads(cache.headsOut[t])
		--Grad w.r.t outW, outB
		for i = 1, #layer.outW do
			dOutB[i] = dOutB[i] + doutSeq[t][i]
			for j = 1, #layer.outW[1] do
				dOutW[i][j] = dOutW[i][j] + doutSeq[t][i] * concat[j]
			end
		end
		--Grad w.r.t concatHeads
		local dConcat = zeros(#concat)
		for j = 1, #concat do
			local s = 0
			for i = 1, #layer.outW do
				s = s + layer.outW[i][j] * doutSeq[t][i]
			end
			dConcat[j] = s
		end
		dConcatHeads[t] = dConcat
	end

	--Backprop Split Heads
	local dHeads = {}
	for t = 1, seqLen do
		dHeads[t] = splitHeads(dConcatHeads[t], heads, dHead)
	end

	--Init dProbs, dScores, dVs, dQs, dKs for attention
	local dAttnProbs = {}
	local dAttnScores = {}
	local dVsH = {}
	local dQsH = {}
	local dKsH = {}
	for t = 1, seqLen do
		dAttnProbs[t] = {}
		dAttnScores[t] = {}
		dVsH[t] = {}
		dQsH[t] = {}
		dKsH[t] = {}
		for h = 1, heads do
			dAttnProbs[t][h] = zeros(seqLen)
			dAttnScores[t][h] = zeros(seqLen)
			dVsH[t][h] = zeros(dHead)
			dQsH[t][h] = zeros(dHead)
			dKsH[t][h] = zeros(dHead)
		end
	end

	--Backprop Attention weighted sum
	for t = 1, seqLen do
		for h = 1, heads do
			for j = 1, seqLen do
				local grad = 0
				for d = 1, dHead do
					grad = grad + doutSeq[t][(h-1)*dHead + d] * cache.attnProbs[t][h][j] --Grad from weighted sum
				end
				--Grad for attnProbs[t][h][j] from weighted sum:
				dAttnProbs[t][h][j] = dAttnProbs[t][h][j] + grad
				--Grad for VsH:
				for d = 1, dHead do
					dVsH[j][h][d] = dVsH[j][h][d] + cache.attnProbs[t][h][j] * doutSeq[t][(h-1)*dHead + d]
				end
			end
		end
	end

	--Backprop Softmax grad (attention scores)
	for t = 1, seqLen do
		for h = 1, heads do
			local probs = cache.attnProbs[t][h]
			local dProb = dAttnProbs[t][h]
			local dScore = zeros(seqLen)
			for i = 1, seqLen do
				local s = 0
				for j = 1, seqLen do
					local delta = (i == j) and 1 or 0
					s = s + dProb[j] * probs[j] * (delta - probs[i])
				end
				dScore[i] = s
			end
			dAttnScores[t][h] = dScore
		end
	end

	--Backprop scores -> Qs, Ks
	for t = 1, seqLen do
		for h = 1, heads do
			local dScore = dAttnScores[t][h]
			for j = 1, seqLen do
				local scale = 1 / math_sqrt(dHead)
				for d = 1, dHead do
					dQsH[t][h][d] = dQsH[t][h][d] + dScore[j] * cache.KsH[j][h][d] * scale
					dKsH[j][h][d] = dKsH[j][h][d] + dScore[j] * cache.QsH[t][h][d] * scale
				end
			end
		end
	end

	--Backprop Ks, Qs, Vs to input x and weight grads
	local dX = {}
	for t = 1, seqLen do
		dX[t] = zeros(dModel)
	end

	local dQW = zerosMatrix(#layer.qW, #layer.qW[1])
	local dKW = zerosMatrix(#layer.kW, #layer.kW[1])
	local dVW = zerosMatrix(#layer.vW, #layer.vW[1])
	local dOutW = dOutW --already defined above
	local dOutB = dOutB

	--combine heads grad to single vector grad
	local function combineHeadsGrad(dHeadsT, heads, dHead)
		local out = {}
		for h = 1, heads do
			for d = 1, dHead do
				table.insert(out, dHeadsT[h][d])
			end
		end
		return out
	end

	for t = 1, seqLen do
		--dQW
		local dxq = zeros(#layer.qW[1])
		local dxk = zeros(#layer.kW[1])
		local dxv = zeros(#layer.vW[1])

		--Sum grads from heads for Q,K,V
		local dQCombined = combineHeadsGrad(dQsH[t], heads, dHead)
		local dKCombined = combineHeadsGrad(dKsH[t], heads, dHead)
		local dVCombined = combineHeadsGrad(dVsH[t], heads, dHead)

		--Compute grads for qW,kW,vW and dx[t]
		for i = 1, #layer.qW do
			for j = 1, #layer.qW[1] do
				dQW[i][j] = dQW[i][j] + dQCombined[i] * cache.xSeq[t][j]
				dKW[i][j] = dKW[i][j] + dKCombined[i] * cache.xSeq[t][j]
				dVW[i][j] = dVW[i][j] + dVCombined[i] * cache.xSeq[t][j]
			end
		end

		for j = 1, #cache.xSeq[t] do
			local s = 0
			for i = 1, #layer.qW do
				s = s + layer.qW[i][j] * dQCombined[i]
			end
			dxq[j] = s

			s = 0
			for i = 1, #layer.kW do
				s = s + layer.kW[i][j] * dKCombined[i]
			end
			dxk[j] = s

			s = 0
			for i = 1, #layer.vW do
				s = s + layer.vW[i][j] * dVCombined[i]
			end
			dxv[j] = s
		end

		--Sum dxq + dxk + dxv
		for j = 1, #cache.xSeq[t] do
			dX[t][j] = dxq[j] + dxk[j] + dxv[j]
		end
	end

	--Return of gradients
	local grads = {
		dQW = dQW,
		dKW = dKW,
		dVW = dVW,
		dOutW = dOutW,
		dOutB = dOutB,
		dX = dX,
	}
	return grads
end

--Tokenizer
function Tokenizer.new()
	local self = setmetatable({}, Tokenizer)
	self.vocab = {}
	self.ivocab = {}
	self.vocabSize = 0

	local special = {
		["<pad>"] = 0,
		["<start>"] = 1,
		["<eos>"] = 2,
		["<unk>"] = 3,
	}
	local index = 4 --After specials

	for token, id in pairs(special) do
		self.vocab[token] = id
		self.ivocab[id] = token
	end

	--ASCII 32 to 126
	for i = 32, 126 do
		local c = string.char(i)
		self.vocab[c] = index
		self.ivocab[index] = c
		index += 1
	end

	self.vocabSize = index --total token count
	return self
end

function Tokenizer:encode(text, addStart, addEos)
	local tokens = {}
	if addStart then table.insert(tokens, self.vocab["<start>"]) end
	for i = 1, #text do
		local c = text:sub(i, i)
		table.insert(tokens, self.vocab[c] or self.vocab["<unk>"])
	end
	if addEos then table.insert(tokens, self.vocab["<eos>"]) end
	return tokens
end

function Tokenizer:decode(tokens)
	local chars = {}
	for i = 1, #tokens do
		local tok = tokens[i]
		local char = self.ivocab[tok]
		if char == "<start>" or char == "<eos>" or char == "<pad>" then
			--Skip these during decode
		else
			table.insert(chars, char or "?")
		end
	end
	return table.concat(chars)
end

--GPT constructor
function GPT.new(opts)
	local self = setmetatable({}, GPT)
	self.dModel = opts.dModel or 64
	self.numLayers = opts.numLayers or 2
	self.heads = opts.heads or 4
	self.ffHidden = opts.ffHidden or 128
	self.vocabSize = opts.vocabSize or 100
	self.maxSeqLen = opts.maxSeqLen or 32

	self.beta1 = 0.9
	self.beta2 = 0.999
	self.eps = 1e-8
	self.step = 0

	--Embeddings
	self.embeddings = randomMatrix(self.vocabSize, self.dModel, 0.1)
	self.positional = randomMatrix(self.maxSeqLen, self.dModel, 0.1)

	--Output Layer
	self.outW = randomMatrix(self.vocabSize, self.dModel, 0.1)
	self.outB = zeros(self.vocabSize)

	--Adam states
	self.adam_embeddings = adamInitMatrix(self.vocabSize, self.dModel)
	self.adam_positional = adamInitMatrix(self.maxSeqLen, self.dModel)
	self.adam_outW = adamInitMatrix(self.vocabSize, self.dModel)
	self.adam_outB = adamInitVector(self.vocabSize)

	--Layers
	self.layers = {}
	for i = 1, self.numLayers do
		local layer = {}
		--Attention weights: q,k,v out
		layer.qW = randomMatrix(self.dModel, self.dModel, 0.1)
		layer.kW = randomMatrix(self.dModel, self.dModel, 0.1)
		layer.vW = randomMatrix(self.dModel, self.dModel, 0.1)
		layer.outW = randomMatrix(self.dModel, self.dModel, 0.1)
		layer.outB = zeros(self.dModel)

		layer.adam_qW = adamInitMatrix(self.dModel, self.dModel)
		layer.adam_kW = adamInitMatrix(self.dModel, self.dModel)
		layer.adam_vW = adamInitMatrix(self.dModel, self.dModel)
		layer.adam_outW = adamInitMatrix(self.dModel, self.dModel)
		layer.adam_outB = adamInitVector(self.dModel)

		--FeedForward
		layer.ffW1 = randomMatrix(self.ffHidden, self.dModel, 0.1)
		layer.ffb1 = zeros(self.ffHidden)
		layer.ffW2 = randomMatrix(self.dModel, self.ffHidden, 0.1)
		layer.ffb2 = zeros(self.dModel)

		layer.adam_ffW1 = adamInitMatrix(self.ffHidden, self.dModel)
		layer.adam_ffb1 = adamInitVector(self.ffHidden)
		layer.adam_ffW2 = adamInitMatrix(self.dModel, self.ffHidden)
		layer.adam_ffb2 = adamInitVector(self.dModel)

		self.layers[i] = layer
	end

	self.tokenizer = Tokenizer.new()
	return self
end

--Forward Pass (Batch: array of token sequences)
function GPT:forwardBatch(inputBatch)
	local batchSize = #inputBatch
	local seqLen = #inputBatch[1]
	local dModel = self.dModel
	local heads = self.heads

	local embeddings = self.embeddings
	local positional = self.positional

	--Token Embeddings + Position
	local xBatch = {}
	for b = 1, batchSize do
		xBatch[b] = {}
		for t = 1, seqLen do
			local token = inputBatch[b][t]
			local tokenEmb = embeddings[token] or zeros(dModel)
			local posEmb = positional[t] or zeros(dModel)
			xBatch[b][t] = vecAdd(tokenEmb, posEmb)
		end
	end

	--Causal Mask for seqLen
	local mask = causalMask(seqLen)

	--For each layer sequentially
	local cacheLayers = {}
	for b = 1, batchSize do
		cacheLayers[b] = {}
	end

	local outBatch = {}
	for b = 1, batchSize do
		local x = xBatch[b]
		for l = 1, self.numLayers do
			local layer = self.layers[l]

			--Multihead Attention Forward
			local attnOut, attnCache = multiHeadAttentionForward(x, layer, heads, dModel, mask)
			--Residual and LayerNorm 1
			local res1 = {}
			for t = 1, seqLen do
				res1[t] = vecAdd(x[t], attnOut[t])
			end
			local ln1Out = {}
			local lnCache = {}
			for t = 1, seqLen do
				local normed, mean, std = layerNorm(res1[t])
				ln1Out[t] = normed
				lnCache[t] = {res1[t], mean, std}
			end

			--FeedForward Forward
			local ffOut = {}
			local ffCache = {}
			for t = 1, seqLen do
				local out, h1 = feedForwardForward(ln1Out[t], layer.ffW1, layer.ffb1, layer.ffW2, layer.ffb2)
				ffOut[t] = out
				ffCache[t] = h1
			end

			--Residual and LayerNorm 2
			local res2 = {}
			local ln2Out = {}
			local ln2Cache = {}
			for t = 1, seqLen do
				res2[t] = vecAdd(ln1Out[t], ffOut[t])
				local normed, mean, std = layerNorm(res2[t])
				ln2Out[t] = normed
				ln2Cache[t] = {res2[t], mean, std}
			end

			--Update x for next layer
			x = ln2Out

			--Save cache
			cacheLayers[b][l] = {
				attnCache = attnCache,
				ln1Cache = lnCache,
				ffCache = ffCache,
				ln2Cache = ln2Cache,
				res1 = res1,
				res2 = res2,
			}
		end
		outBatch[b] = x
	end

	--Output Layer (Logits)
	local logitsBatch = {}
	for b = 1, batchSize do
		logitsBatch[b] = {}
		for t = 1, seqLen do
			local logit = vecAdd(matVecMul(self.outW, outBatch[b][t]), self.outB)
			logitsBatch[b][t] = logit
		end
	end

	return logitsBatch, cacheLayers, inputBatch
end

--Backpropagation Batch
function GPT:backpropBatch(cacheLayers, inputBatch, targetBatch, lr)
	local batchSize = #inputBatch
	local seqLen = #inputBatch[1]
	local dModel = self.dModel
	local heads = self.heads
	local ffHidden = self.ffHidden
	local dHead = dModel / heads

	--Init gradients for embeddings, positional, output layer
	local gradEmbeddings = zerosMatrix(self.vocabSize, dModel)
	local gradPositional = zerosMatrix(self.maxSeqLen, dModel)
	local gradOutW = zerosMatrix(self.vocabSize, dModel)
	local gradOutB = zeros(self.vocabSize)

	--Grads per layer
	local gradLayers = {}
	for l = 1, self.numLayers do
		gradLayers[l] = {
			dqW = zerosMatrix(dModel, dModel),
			dkW = zerosMatrix(dModel, dModel),
			dvW = zerosMatrix(dModel, dModel),
			doutW = zerosMatrix(dModel, dModel),
			doutB = zeros(dModel),

			dffW1 = zerosMatrix(ffHidden, dModel),
			dffb1 = zeros(ffHidden),
			dffW2 = zerosMatrix(dModel, ffHidden),
			dffb2 = zeros(dModel),
		}
	end

	--Gradient Output Layer (CrossEntropy)
	local dOutLayer = {}
	for b = 1, batchSize do
		dOutLayer[b] = {}
		for t = 1, seqLen do
			local logits = cacheLayers[b][self.numLayers].ln2Cache[t][1] --last layer output before output layer
			local pred = softmax(vecAdd(matVecMul(self.outW, logits), self.outB))
			local gradLogits = {}
			for v = 1, self.vocabSize do
				gradLogits[v] = pred[v]
			end
			gradLogits[targetBatch[b][t]] = gradLogits[targetBatch[b][t]] - 1
			dOutLayer[b][t] = gradLogits
		end
	end

	--Backprop Output Layer Parameters and Input Grad
	local dOutLayerInput = {}
	for b = 1, batchSize do
		dOutLayerInput[b] = {}
		for t = 1, seqLen do
			local dout = dOutLayer[b][t]
			local ln2Out = cacheLayers[b][self.numLayers].ln2Cache[t][1]

			--Grad w.r.t outW, outB
			for i = 1, self.vocabSize do
				gradOutB[i] = gradOutB[i] + dout[i]
				for j = 1, dModel do
					gradOutW[i][j] = gradOutW[i][j] + dout[i] * ln2Out[j]
				end
			end

			--Grad w.r.t ln2 output (Input for Layer N)
			local dln2 = zeros(dModel)
			for j = 1, dModel do
				local s = 0
				for i = 1, self.vocabSize do
					s = s + self.outW[i][j] * dout[i]
				end
				dln2[j] = s
			end

			dOutLayerInput[b][t] = dln2
		end
	end

	--Backprop through layers (from layer N to layer 1)
	--dLayerInput = dOutLayerInput (input Grad of output layer)
	local dLayerInput = dOutLayerInput

	for l = self.numLayers, 1, -1 do
		local grad = gradLayers[l]
		local layer = self.layers[l]

		--Grad for next layer input (for residual connection)
		local dNextInput = {}

		for b = 1, batchSize do
			dNextInput[b] = {}
		end

		--Per Batch, Per Time (seqLen)
		for b = 1, batchSize do
			local cache = cacheLayers[b][l]
			local ln2Out = cache.ln2Cache 		--LayerNorm 2 Cache (normed output)
			local res2 = cache.res2 			--Input for LayerNorm 2 (ln1Out + ffOut)
			local ffCache = cache.ffCache 		--Interim result FFN h1
			local ln1Out = cache.ln1Cache 		--LayerNorm 1 Cache (normed output)
			local res1 = cache.res1 			--Input for LayerNorm 1 (x + attnOut)
			local attnCache = cache.attnCache	--Attention Cache (for Backprop)

			for t = 1, seqLen do
				--LayerNorm 2 Backprop
				local dln2Out = dLayerInput[b][t] --Gradient of output layer / next layer

				--LayerNorm backward: Input grad and parameter grad (LayerNorm has no parameters here)
				local dres2 = layerNormBackward(dln2Out, res2[t], ln2Out[t][2], ln2Out[t][3])

				--Residual Add (ln1Out + ffOut)
				--dres2 is gradient on sum of ln1Out + ffOut
				local dln1Out = zeros(dModel)
				local dffOut = zeros(dModel)
				for i = 1, dModel do
					--Pass gradient to both paths
					dln1Out[i] = dres2[i]
					dffOut[i] = dres2[i]
				end

				--FeedForward Backprop
				--FF Forward:
				--h1 = relu(W1*x + b1)
				--out = W2*h1 + b2
				local h1 = ffCache[t]

				--Backprop out = W2*h1 + b2
				local dffb2 = dffOut
				local dffW2 = zerosMatrix(dModel, ffHidden)
				local dh1 = zeros(ffHidden)
				for i = 1, dModel do
					for j = 1, ffHidden do
						dffW2[i][j] = dffW2[i][j] + dffOut[i] * h1[j]
						dh1[j] = dh1[j] + layer.ffW2[i][j] * dffOut[i]
					end
				end

				--Backprop h1 = relu(W1*x + b1)
				local dh1raw = zeros(ffHidden)
				for i = 1, ffHidden do
					dh1raw[i] = (h1[i] > 0) and dh1[i] or 0
				end

				local dffb1 = dh1raw
				local dffW1 = zerosMatrix(ffHidden, dModel)
				local dxFF = zeros(dModel)
				for i = 1, ffHidden do
					for j = 1, dModel do
						dffW1[i][j] = dffW1[i][j] + dffb1[i] * ln1Out[t][j]
						dxFF[j] = dxFF[j] + layer.ffW1[i][j] * dffb1[i]
					end
				end

				--Residual Add (x + attnOut)
				--dln1Out comes from above
				local dres1 = zeros(dModel)
				for i = 1, dModel do
					dres1[i] = dln1Out[i] + dxFF[i]
				end

				--LayerNorm 1 Backprop
				local dln1 = layerNormBackward(dres1, res1[t], ln1Out[t][2], ln1Out[t][3])

				--MultiHead Attention Backprop
				local attnGrads = multiHeadAttentionBackward(dln1, attnCache, layer, heads, dModel)

				--multiHeadAttentionBackward gives:
				--{
				--  dqW, dkW, dvW,
				--  doutW, doutB,
				--  dX: gradient input x for next layer
				--  dQ, dK, dV
				--}

				--Summing gradients to the layer level
				grad.dqW = matAdd(grad.dqW, attnGrads.dqW)
				grad.dkW = matAdd(grad.dkW, attnGrads.dkW)
				grad.dvW = matAdd(grad.dvW, attnGrads.dvW)
				grad.doutW = matAdd(grad.doutW, attnGrads.doutW)
				grad.doutB = vecAdd(grad.doutB, attnGrads.doutB)

				--Sum of gradients on residual input for the previous layer
				dNextInput[b][t] = attnGrads.dX
			end
		end

		dLayerInput = dNextInput
	end

	--Calculate embeddings and positional degrees (via InputBatch and dLayerInput)
	for b = 1, batchSize do
		local inputSeq = inputBatch[b]
		for t = 1, seqLen do
			local token = inputSeq[t]
			local dvec = dLayerInput[b][t]

			if token > 0 then
				for j = 1, dModel do
					gradEmbeddings[token][j] = gradEmbeddings[token][j] + dvec[j]
				end
			end
			for j = 1, dModel do
				gradPositional[t][j] = gradPositional[t][j] + dvec[j]
			end
		end
	end

	--Parameter update with Adam
	self.step = self.step + 1
	local lrAdam = lr

	adamUpdateParam(self.embeddings, gradEmbeddings, self.adam_embeddings, lrAdam, self.beta1, self.beta2, self.eps)
	adamUpdateParam(self.positional, gradPositional, self.adam_positional, lrAdam, self.beta1, self.beta2, self.eps)

	adamUpdateParam(self.outW, gradOutW, self.adam_outW, lrAdam, self.beta1, self.beta2, self.eps)
	adamUpdateParam(self.outB, gradOutB, self.adam_outB, lrAdam, self.beta1, self.beta2, self.eps)

	for l = 1, self.numLayers do
		local grad = gradLayers[l]
		local layer = self.layers[l]

		adamUpdateParam(layer.qW, grad.dqW, layer.adam_qW, lrAdam, self.beta1, self.beta2, self.eps)
		adamUpdateParam(layer.kW, grad.dkW, layer.adam_kW, lrAdam, self.beta1, self.beta2, self.eps)
		adamUpdateParam(layer.vW, grad.dvW, layer.adam_vW, lrAdam, self.beta1, self.beta2, self.eps)

		adamUpdateParam(layer.outW, grad.doutW, layer.adam_outW, lrAdam, self.beta1, self.beta2, self.eps)
		adamUpdateParam(layer.outB, grad.doutB, layer.adam_outB, lrAdam, self.beta1, self.beta2, self.eps)

		adamUpdateParam(layer.ffW1, grad.dffW1, layer.adam_ffW1, lrAdam, self.beta1, self.beta2, self.eps)
		adamUpdateParam(layer.ffb1, grad.dffb1, layer.adam_ffb1, lrAdam, self.beta1, self.beta2, self.eps)
		adamUpdateParam(layer.ffW2, grad.dffW2, layer.adam_ffW2, lrAdam, self.beta1, self.beta2, self.eps)
		adamUpdateParam(layer.ffb2, grad.dffb2, layer.adam_ffb2, lrAdam, self.beta1, self.beta2, self.eps)
	end
end

--Training function (simple not complex may break)
function GPT:train(data, epochs, batchSize, lr)
	local tokenizer = self.tokenizer
	local maxSeqLen = self.maxSeqLen

	for epoch = 1, epochs do
		print("Epoch " .. epoch .. " start")

		local i = 1
		local totalLoss = 0
		local totalTokens = 0
		local correctTokens = 0

		while i <= #data do
			local batch = {}
			local targetBatch = {}

			for b = 1, batchSize do
				local idx = i + b - 1
				if idx > #data then break end

				--Encode with <start> and <eos>
				local tokens = tokenizer:encode(data[idx], true, true)

				--Trim if longer than max sequence
				if #tokens > maxSeqLen then
					table.remove(tokens) --remove <eos> to fit length
				end

				--Prepare shifted input/target
				local inputSeq = {}
				local targetSeq = {}
				for t = 1, (#tokens - 1) do
					inputSeq[t] = tokens[t]
					targetSeq[t] = tokens[t + 1]
				end

				--Pad to maxSeqLen
				while #inputSeq < maxSeqLen do
					table.insert(inputSeq, tokenizer.vocab["<pad>"])
				end
				while #targetSeq < maxSeqLen do
					table.insert(targetSeq, tokenizer.vocab["<pad>"])
				end

				--Cut off if too long
				while #inputSeq > maxSeqLen do
					table.remove(inputSeq)
				end
				while #targetSeq > maxSeqLen do
					table.remove(targetSeq)
				end

				table.insert(batch, inputSeq)
				table.insert(targetBatch, targetSeq)
			end

			if #batch == 0 then break end

			local logitsBatch, cache, inputs = self:forwardBatch(batch)
			self:backpropBatch(cache, inputs, targetBatch, lr)

			--Compute batch loss and accuracy
			for b = 1, #batch do
				for t = 1, maxSeqLen do
					local targetToken = targetBatch[b][t]
					if targetToken ~= tokenizer.vocab["<pad>"] then
						local logits = logitsBatch[b][t]
						local predProb = softmax(logits)
						local targetProb = predProb[targetToken] or 1e-10
						totalLoss = totalLoss - math.log(targetProb + 1e-10)

						--Get argmax prediction
						local maxProb = -math.huge
						local predToken = 0
						for v = 1, #predProb do
							if predProb[v] > maxProb then
								maxProb = predProb[v]
								predToken = v
							end
						end

						if predToken == targetToken then
							correctTokens += 1
						end
						totalTokens += 1
					end
				end
			end

			i += batchSize
		end

		local avgLoss = totalLoss / math.max(1, totalTokens)
		local accuracy = totalTokens > 0 and (correctTokens / totalTokens) or 0

		print(string.format("Epoch %d done - Loss: %.4f, Accuracy: %.4f", epoch, avgLoss, accuracy))
	end
end

return GPT
