function [ Weight, Bias, InputOutput, SignalFlow ] = NeuronBlock( SignalFlow, Weight, Bias, InputOutput, FunctionType, Slope_Parameter, FunctionAmplitude )
%This function performs processing of block and updates its structure,
%weights, and biases in the case of structre-changing croosover operation 
%and sends back new biases and weights of the block.
%Also it updates InputOutput matrix by replacing new outputs on old ones.
%
%Note: In SignalFlow matrix "0" means input and "1" means output.

%%%updating weights
%%%Note: In restructuring phase, weight 1,2 never changes.
%weight 1,3
if ((SignalFlow(1)~=SignalFlow(3))&&(SignalFlow(1)==1))
    Weight(2)=random('normal',0,1);
elseif ((SignalFlow(1)~=SignalFlow(3))&&(SignalFlow(1)==0))
    Weight(2)=0;
end

%weight 1,4
if ((SignalFlow(2)~=SignalFlow(4))&&(SignalFlow(2)==1))
    Weight(3)=random('normal',0,1);
elseif ((SignalFlow(2)~=SignalFlow(4))&&(SignalFlow(2)==0))
    Weight(3)=0;
end

%weight 2,3
if ((SignalFlow(1)~=SignalFlow(3))&&(SignalFlow(1)==1))
   Weight(4)=0;
elseif ((SignalFlow(1)~=SignalFlow(3))&&(SignalFlow(1)==0))
   Weight(4)=random('normal',0,1);
end

%weight 2,4
if ((SignalFlow(2)~=SignalFlow(4))&&(SignalFlow(2)==1))
    Weight(5)=0;
elseif ((SignalFlow(2)~=SignalFlow(4))&&(SignalFlow(2)==0))
    Weight(5)=random('normal',0,1);
end

%weight 3,4
if (or((SignalFlow(2)~=SignalFlow(4)),(SignalFlow(1)~=SignalFlow(3)))&&(SignalFlow(1)~=SignalFlow(2)))
    Weight(6)=random('normal',0,1);
elseif (or((SignalFlow(2)~=SignalFlow(4)),(SignalFlow(1)~=SignalFlow(3)))&&(SignalFlow(1)==SignalFlow(2)))
    Weight(6)=0;
end

%%%updating biases
%%%Note: In restructuring phase, bias 2 never changes because neuron 2 is
%%%always an output.
%bias 3
if ((SignalFlow(1)~=SignalFlow(3))&&(SignalFlow(1)==1))
    Bias(2)=random('normal',0,1);
elseif ((SignalFlow(1)~=SignalFlow(3))&&(SignalFlow(1)==0))
    Bias(2)=0;
end

%bias 4
if ((SignalFlow(2)~=SignalFlow(4))&&(SignalFlow(2)==1))
    Bias(3)=random('normal',0,1);
elseif ((SignalFlow(2)~=SignalFlow(4))&&(SignalFlow(2)==0))
    Bias(3)=0;
end

%%%updating outputs
%output2
InputOutput(2)=ActivationFunction(((Weight(1)*InputOutput(1))+(Weight(4)*InputOutput(3))+(Weight(5)*InputOutput(4))+Bias(1)),FunctionType,Slope_Parameter, FunctionAmplitude);

%output3
if SignalFlow(1)==1
    InputOutput(3)=ActivationFunction(((Weight(2)*InputOutput(1))+(Weight(6)*InputOutput(4))+Bias(2)),FunctionType,Slope_Parameter, FunctionAmplitude);
end

%output4
if SignalFlow(2)==1
    InputOutput(4)=ActivationFunction(((Weight(3)*InputOutput(1))+(Weight(6)*InputOutput(3))+Bias(3)),FunctionType,Slope_Parameter, FunctionAmplitude);
end

%%%saving signal flow state
SignalFlow(:,:,:,3)=SignalFlow(:,:,:,1);
SignalFlow(:,:,:,4)=SignalFlow(:,:,:,2);

end
