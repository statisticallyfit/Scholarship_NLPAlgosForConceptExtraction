def forward(self, hidden: Tensor, encoderOutputs: Tensor) -> Tensor:
    """
    
    :param hidden:  
        shape = (batchSize, decoderHiddenDim) 
    :param encoderOutputs: 
        shape = (srcSentenceLen, batchSize, encoderHiddenDim * 2)
    :return: 
    """
    srcSentenceLen, batchSize, _ = encoderOutputs.shape 
    
    # repeat encoder hidden state srcSentenceLen times
    hidden: Tensor = hidden.unsqueeze(1).repeat(1, srcSentenceLen, 1)
    ## hidden shape = (batchSize, srcSentenceLen, decoderHiddenDim)
    
    encoderOutputs: Tensor = encoderOutputs.permute(1, 0, 2)
    ## encoderOutputs shape = (batchSize, srcSentenceLen, encoderHiddenDim * 2)
    
    energy: Tensor = torch.tanh(self.attentionLinearLayer(
        torch.cat((hidden, encoderOutputs), dim = 2)))
    ## energy shape = (batchSize, srcSentenceLen, decoderHiddenDim)
    
    energy: Tensor = energy.permute(0, 2, 1)
    ## energy shape now = (batchSize, decoderHiddenDim, srcSentenceLen)
    
    # V shape = (decoderHiddenDim)
    # NOTE: v has aesthetic purpose only, to properly shape the attention vector.
    v: Tensor = self.v.repeat(batchSize, 1).unsqueeze(1)
    ## v shape now = (batchSize, 1, decoderHiddenDim)
    
    # Multiplying v * E_t then removing the dimension 1 to get attention vector
    # of shape (srcSentenceLen)
    attention: Tensor = torch.bmm(v, energy).squeeze(1) 
    ## attention shape = (batchSize, srcLen)
    # TODO: srcLen == srcSentenceLen ????
    # TODO: why is this not the expected shape? (vector)?
    
    return F.softmax(input = attention, dim = 1)