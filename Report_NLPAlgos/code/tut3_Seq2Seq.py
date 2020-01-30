class Seq2Seq(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        
        super().__init__()
        
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.device = device
        
        
        
    def forward(self, srcSeq: Tensor, trgSeq: Tensor,
                teacherForcingRatio = 0.5) -> Tensor:
        """
        
        :param srcSeq:
            shape = (srcSentenceLen, batchSize) 
        :param trgSeq: 
            shape = (trgSentenceLen, batchSize)
        :param teacherForcingRatio: 
        
        :return: 
        """
        maxLen, batchSize = trgSeq.shape 
        trgVocabSize: int = self.decoder.outputDim 
        
        # tensor to store decoder outputs
        outputs: Tensor = torch.zeros(maxLen, batchSize, trgVocabSize).to(self.device)
        
        # Encoder outputs is all the hidden states of the input sequence, 
        # backwards and forwads
        ## Hidden is final forward and backward hidden states, after having passed
        # through a linear layer
        encoderOutputs, hidden = self.encoder(srcSeq)
        
        # First input to the decoder is the <sos> tokens
        input = trgSeq[0, :]
        
        for t in range(1, maxLen):
            # insert input token embedding, previous hidden state, and all
            # encoder hidden states
            # Receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input = input, hidden=hidden, 
                                          encoderOutputs = encoderOutputs)
            
            # Place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # Decide if using teacher forcing or not
            useTeacherForce = random.random() < teacherForcingRatio
            
            # Get highest predicted token from our predictions
            maxPredToken = output.argmax(1)
            
            # if teacher forcing, use actual next token (trgSeq[t]) as next input, 
            # else use predicted token (maxPredToken)
            input = trgSeq[t] if useTeacherForce else maxPredToken
            
            
        return outputs