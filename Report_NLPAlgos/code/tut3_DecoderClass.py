class Decoder(nn.Module):
    
    def __init__(self, outputDim: int, embedDim: int, 
                 encoderHiddenDim: int, decoderHiddenDim: int, dropout: float, 
                 attention: Attention):
        
        super().__init__()
        
        self.outputDim: int = outputDim 
        self.attention: Attention = attention
        
        self.embeddingLayer = nn.Embedding(num_embeddings=outputDim,
                                           embedding_dim=embedDim)
        
        self.rnn = nn.GRU(input_size= encoderHiddenDim * 2 + embedDim,
                          hidden_size= decoderHiddenDim)
        
        self.outputLayer = nn.Linear(
            in_features=encoderHiddenDim * 2 + decoderHiddenDim + embedDim,
            out_features=outputDim)
        
        self.dropout = nn.Dropout(dropout)
        
        
        
    def forward(self, input:Tensor, hidden: Tensor, 
                encoderOutputs: Tensor) -> (Tensor, Tensor):
        """
        
        :param input: 
            shape = (batchSize)
        :param hidden: 
            shape = (batchSize, decoderHiddenDim)
        :param encoderOutputs: 
            shape = (srcSentenceLen, batchSize, encoderHiddenDim * 2)
            
        :return: 
        """
        
        input: Tensor = input.unsqueeze(0)
        ## input shape now = (1, batchSize)
        
        inputEmbedding: Tensor = self.dropout(self.embeddingLayer(input))
        ## shape = (1, batchSize, embedDim)
        
        # Calculate attention (result of forward method)
        a: Tensor = self.attention(hidden=hidden, encoderOutputs = encoderOutputs)
        ## a shape = (batchSize, srcLen)
        a: Tensor = a.unsqueeze(1) # add 1-dim tensor at dim = 1
        ## a size = (batchSize, 1, srcLen)
        
        encoderOutputs: Tensor = encoderOutputs.permute(1, 0, 2)
        ## shape = (batchSize, srcSentenceLen, encoderHiddenDim * 2)
        ## NOTE: meaning of torch.permute(): https://hyp.is/vV91khFMEeqUaP99rnuMeg/kite.com/python/docs/torch.Tensor.permute
        # torch.permute() switches the dimensions of the tensor by referring to the axes / dims
        
        # called 'weighted'
        ## NOTE: meaning of torch.bmm(): https://hyp.is/XqvTdBFMEeqVmoOfrqLXCA/pytorch.org/docs/stable/torch.html
        # Meaning: does batch matrix multiplication: 
        #   if mat1 has size (b, n, m) and mat2 has size (b, m, p), the result of 
        #   torch.bmm(mat1, mat2) is a matrix with size (b, n, p)
        weightedContext: Tensor = torch.bmm(a, encoderOutputs)
        # weighted context shape = (batchSize, 1, encoderHiddenDim * 2)
        weightedContext: Tensor = weightedContext.permute(1, 0, 2)
        ## shape = (1, batchSize, encoderHiddenDim * 2)
        
        rnnInput: Tensor = torch.cat((inputEmbedding, weightedContext), dim = 2)
        ## shape = (1, batchSize, encoderHiddenDIm * 2 + embedDim)
        
        #output, hidden = self.rnn(input = rnnInput, hidden = hidden.unsqueeze(0))
        output, hidden = self.rnn(rnnInput, hidden.unsqueeze(0))
        ## output shape = (sentenceLen, batchSize, decoderHiddenDim * numDirections)
        ## hidden shape = (numLayers * numDirections, batchSize, decoderHiddenDim)
        ## note: sentenceLen = numLayers = numDirections = 1, so the shapes are:
        ## output shape = (1, batchSize, decoderHiddenDim)
        ## hidden shape = (1, batchSize, decoderHiddenDim)
        ## TODO (?): this also means that output == hidden
        # why are they equal?
        assert (output == hidden).all()
        
        
        # Getting rid of 1-dimensional tensor for all of these
        inputEmbedding: Tensor = inputEmbedding.squeeze(0)
        ## shape = (batchSize, embedDim)
        output: Tensor = output.squeeze(0)
        ## shape = (batchSize, decoderHiddenDim)
        weightedContext: Tensor = weightedContext.squeeze(0)
        ## shape = (batchSize, encoderHiddenDim * 2)
        
        prediction: Tensor = self.outputLayer(
            torch.cat( (output, weightedContext, inputEmbedding), dim=1 )
        )
        # outut shape = (batchSize, outputDim)
        
        hidden: Tensor = hidden.squeeze(0) # put back 1-dim tensor at position 0
        # hidden shape = (1, 1, batchSize, decoderHiddemDim) 
        # TODO; is this correct?
        
        
        return prediction, hidden