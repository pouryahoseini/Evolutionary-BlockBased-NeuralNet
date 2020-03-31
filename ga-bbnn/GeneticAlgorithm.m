function [ Best_SignalFlow_Matrix, Best_Weight_Matrix, Best_Bias_Matrix, Best_Fitness ] = GeneticAlgorithm( Example_Matrix, NetworkRows, TerminationRound, Population, TournamentSize, DeletionNumber, CrossoverRate, Structure_MutationRate, WeightBias_MutationRate, FunctionType, Slope_Parameter, FunctionAmplitude, ComparisonType, OutputNumber )
%This function carries out evolutionary training of the neural network. The
%implemented genetic algorithm is on the steady state basis with delete the
%worst sterategy, which eliminates "DeletionNumber" of most weak
%chromosomes in each iteration of algorithm. Default deletion number is 2.
%
%It generates a random population with the size of "Population", and then
%performs the genetic operators on them. Default population size is 100.
%
%Tournament selection have been chosen for selection phase with the size of
%"TournamentSize", in which best chromosome in tournament will win and be
%selected as one of the parents. Default tournament size is 4.
%
%Crossover(2D uniform crossover) operator selects two signal flows and with
%a probability of "CrossoverRate" swaps two corresponding connections of 
%them or just balances two corresponding weights to a moderate value based
%on a random number. Default crossover rate is 0.9.
%
%Mutation (2D one point mutation) operator inverts a connection in the 
%selected signal flow and/or simply changes a weight with a probability of 
%"MutationRate". Default mutation rate for structures is 0.03 and for 
%weights and biases is 0.05. Mutation over weights and biases only performs
%one mutation operation on a randomly selected weight or bias.
%
%This function executes the genetic algorithm a predefined nember of
%iterations. It is set by "TerminationRound" entery. Default number for
%termination round is 200.
%
%In addition to run genetic algorithm, this function initializes the neural
%network and generates its structure according to the "NetworkRow" and
%"Example_Matrix". "NetworkRow" defines the layers of the neural network
%and half of the column size of "Example_Matrix" specifies the number of
%inputs and outputs of the network. Default number for network rows is 3.
%
%Note: Each row in Example_Matrix stands for an example, and first half
%columns represent inputs of the example and second half columns indicate
%output values. 
%
%Input "FunctionType" determines the type of activation function used for
%processing of neurons. Default type is sigmoid function with slope
%parameter of 2. If the entered string by user differs from 'sigmoid' and
%be 'gaussian', a gaussian function with the specified slope parameter and
%in other cases a sign function will be used for neurons processing.
%Also amplitude (vertical magnitude in transfer functions) of the selected
%activation function can be changed according to specific problems. It's 1
%by default.
%
%Input "ComparisonType" specifies the way that error (and so fitness) is
%generated. If the entered value be 'numeric', before fitness calculation, 
%outputs of neural network and also examples convert to decimal number and
%then will be compared with each other. In this mode left bit of output
%strings are most significant bit. If the entered value for
%"ComparisonType" be different from 'numeric', fitness calculation will be
%bitwise. In this mode, NN's output bits are compared with corresponding 
%bits of example bits and the sum of differences is error.
%Note: Both numeric and bitwise fitness calculations can be used in the
%case of utilizing sigmoid function as activation function. To see how they
%works with sigmoid activation functions please see FitnessFunction file.
%
%The argument "OutputNumber" tells the program what outputs are considered
%as computation material. This is usefull when only part of outputs of
%neural network or examples are needed. The "OutputNumber" is a vector
%which every value of it represents an output number from the left.

%clear screen
clc;

%set random stream
get_clock=clock;
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(25000000*get_clock(4:6))));

%%%check for number of inputs and outputs
error(nargchk(1,14,nargin));
error(nargchk(3,4,nargout));

%%%set default values of inputs
switch nargin
    case 1
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        FunctionType='sigmoid';
        Slope_Parameter=2;
        Structure_MutationRate=0.03;
        WeightBias_MutationRate=0.05;
        CrossoverRate=0.9;
        DeletionNumber=2;
        TournamentSize=4;
        Population=100;
        TerminationRound=200;
        NetworkRows=3;
    case 2
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        FunctionType='sigmoid';
        Slope_Parameter=2;
        Structure_MutationRate=0.03;
        WeightBias_MutationRate=0.05;
        CrossoverRate=0.9;
        DeletionNumber=2;
        TournamentSize=4;
        Population=100;
        TerminationRound=200;
    case 3
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        FunctionType='sigmoid';
        Slope_Parameter=2;
        Structure_MutationRate=0.03;
        WeightBias_MutationRate=0.05;
        CrossoverRate=0.9;
        DeletionNumber=2;
        TournamentSize=4;
        Population=100;
    case 4
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        FunctionType='sigmoid';
        Slope_Parameter=2;
        Structure_MutationRate=0.03;
        WeightBias_MutationRate=0.05;
        CrossoverRate=0.9;
        DeletionNumber=2;
        TournamentSize=4;
    case 5
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        FunctionType='sigmoid';
        Slope_Parameter=2;
        Structure_MutationRate=0.03;
        WeightBias_MutationRate=0.05;
        CrossoverRate=0.9;
        DeletionNumber=2;
    case 6
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        FunctionType='sigmoid';
        Slope_Parameter=2;
        Structure_MutationRate=0.03;
        WeightBias_MutationRate=0.05;
        CrossoverRate=0.9;
    case 7
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        FunctionType='sigmoid';
        Slope_Parameter=2;
        Structure_MutationRate=0.03;
        WeightBias_MutationRate=0.05;
    case 8
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        FunctionType='sigmoid';
        Slope_Parameter=2;
        WeightBias_MutationRate=0.05;
    case 9
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        FunctionType='sigmoid';
        Slope_Parameter=2;
    case 10
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
        Slope_Parameter=2;
    case 11
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
        FunctionAmplitude=1;
    case 12
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
        ComparisonType='numeric';
    case 13
        OutputNumber=1:numel(Example_Matrix(1,:))/2;
end

%%%check for active outputs (OutputNumber) be less than total outputs
if numel(OutputNumber)>numel(Example_Matrix(1,:))
    error('Active outputs ("OutputNumber") must be equal or less than outputs');
end

%%%check for DeletionNumber to be even and less than population number
if mod(DeletionNumber,2)~=0
    error('DeletionNumber must be even. Number of new replacing offspring is dependent on the number of parents which it should be even for proper crossover operation.');
elseif DeletionNumber>=Population
    error('DeletionNumber is larger than the specified population number.');
elseif DeletionNumber<=0
    error('DeletionNumber is equal or less than zero');
end

%%%set network's number of inputs and outputs
Example_Number=numel(Example_Matrix(:,1));

if mod(numel(Example_Matrix(1,:)),2)~=0
    error('Matrix of examples must have even columns because it contains 2*n columns where n is input/output number of the neural network');
end

NetworkColumns=(numel(Example_Matrix(1,:)))/2;

%%%initialize neural network0.1
Main_Weight_Population=random('normal',0,1,[Population,NetworkRows,NetworkColumns,6]);
Main_Bias_Population=random('normal',0,1,[Population,NetworkRows,NetworkColumns,3]);

%%%initialize population

%reset best founds
Best_SignalFlow_Matrix=zeros(NetworkRows,NetworkColumns,2);
Best_Fitness=0;

%set all signal flows to -1. This causes that all blocks detect a change in
%their signal flows and so generate new weights and biases.
Main_SignalFlow_Population=zeros(Population,NetworkRows,NetworkColumns,4)-1;

%reset fitness matrix
Main_Fitness_Matrix=zeros(1,Population);

for population_counter=1:Population
    %display chromosome counter
    clc;
    disp(['Generating chromosome number: ',num2str(population_counter)]);
    
    %randomize odd columns' signal flows
    Main_SignalFlow_Population(population_counter,1:NetworkRows,1:2:NetworkColumns,1:2)=randi(2,[1,NetworkRows,floor((NetworkColumns+1)/2),2])-1;
    
    %specify even columns' signal flows based on their master neighbor odd
    %columns
    row_counter=1:NetworkRows;
    even_column_counter=2:2:NetworkColumns;
    Main_SignalFlow_Population(population_counter,row_counter,even_column_counter,1)=abs(1-Main_SignalFlow_Population(population_counter,row_counter,even_column_counter-1,2));
    Main_SignalFlow_Population(population_counter,row_counter,even_column_counter,2)=abs(1-Main_SignalFlow_Population(population_counter,row_counter,min(even_column_counter+1,NetworkColumns),1));
    
    %set both right and left of nural network as output
    Main_SignalFlow_Population(population_counter,1:NetworkRows,1,1)=1;
    Main_SignalFlow_Population(population_counter,1:NetworkRows,NetworkColumns,2)=1;

    %fitness estimation of the new individual
    [Main_Fitness_Matrix(1,population_counter), Main_Weight_Population(population_counter,:,:,:), Main_Bias_Population(population_counter,:,:,:), Main_SignalFlow_Population(population_counter,:,:,:)] = FitnessFunction(Example_Matrix, Example_Number, NetworkRows, NetworkColumns, Main_SignalFlow_Population(population_counter,:,:,:), Main_Weight_Population(population_counter,:,:,:), Main_Bias_Population(population_counter,:,:,:), FunctionType, Slope_Parameter, FunctionAmplitude, ComparisonType, OutputNumber);

    %check for best found
    if Main_Fitness_Matrix(1,population_counter)>=Best_Fitness
        Best_Fitness=Main_Fitness_Matrix(1,population_counter);
        Best_SignalFlow_Matrix=Main_SignalFlow_Population(population_counter,:,:,:);
        Best_Weight_Matrix=Main_Weight_Population(population_counter,:,:,:);
        Best_Bias_Matrix=Main_Bias_Population(population_counter,:,:,:);
    end

end

%%%start genetic algorithm iterations
for iteration=1:TerminationRound
    %display iteration number
    clc;
    disp(['Iteration: ',num2str(iteration)]);
    
    %%%selection
    Winner_Index_Vector=zeros(1, DeletionNumber);
    
    for parent_counter=1:DeletionNumber
        %reset winner fitness to 0.
        %Note: Fitness values are positive numbers.
        winner_fitness=0;
        
        %set an initial winner so it prevents failures in case of all zero
        %fitness contestants
        winner_index=1;
        
        for contestant_counter=1:TournamentSize
            %select a random signal flow
            contestant_index=randi(Population);
            
            %call fitness of selected signal flow
            Fitness=Main_Fitness_Matrix(1,contestant_index);
                  
            %check for superiority of selected signal flow
            if Fitness>winner_fitness
                winner_fitness=Fitness;
                winner_index=contestant_index;
            end
        end
        
        Winner_Index_Vector(parent_counter)=winner_index;
    end
    
    %copying population matrices to save original population data during
    %mating operations
    SignalFlow_Population=Main_SignalFlow_Population;
    Weight_Population=Main_Weight_Population;
    Bias_Population=Main_Bias_Population;
    Fitness_Matrix=Main_Fitness_Matrix;

    %%%crossover
    %Note: This section adjusts SignalFlow_Population, Weight_Population,
    %and Bias_Population
    if rand<=CrossoverRate
        for crossover_operation_counter=1:2:DeletionNumber
        
            %perform uniform crossover in the 2D space
            for row_counter=1:NetworkRows
                for column_counter=2:NetworkColumns
                    if rand<=0.5
                        %check if selected signal flows are equal and take
                        %a decision
                        if SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter,1)~=SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,1)
                            %swap signal flows at the left side of selected
                            %blocks
                            Signal_Stash=SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter,1);
                            SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter,1)=SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,1);
                            SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,1)=Signal_Stash;
                            
                            %commit swapping on the previous column's right
                            %side signal flows
                            Signal_Stash=SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter-1,2);
                            SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter-1,2)=SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter-1,2);
                            SignalFlow_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter-1,2)=Signal_Stash;
                            
                        else
                            %change corresponding weights and biases at the
                            %two side of signal flow
                            
                            %left side of signal flow
                            %weights
                            %assign weights of a block connected to its
                            %right input/output
                            if (Weight_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter-1,6)==0)||(Weight_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter-1,6)==0)
                                left_block_weights=[3,5];
                            else
                                left_block_weights=[3,5,6];
                            end
                            
                            %perform weight balance crossover
                            balance_factor=rand;
                            weight_stash=Weight_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter-1,left_block_weights);
                            Weight_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter-1,left_block_weights)=(balance_factor*weight_stash)+((1-balance_factor)*Weight_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter-1,left_block_weights));
                            Weight_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter-1,left_block_weights)=((1-balance_factor)*weight_stash)+(balance_factor*Weight_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter-1,left_block_weights));
                            
                            %biases
                            balance_factor=rand;
                            bias_stash=Bias_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter-1,3);
                            Bias_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter-1,3)=(balance_factor*bias_stash)+((1-balance_factor)*Bias_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter-1,3));
                            Bias_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter-1,3)=((1-balance_factor)*bias_stash)+(balance_factor*Bias_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter-1,3));
                            
                            %right side of signal flow
                            %weights
                            %assign weights of a block connected to its
                            %left input/output
                            if (Weight_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter,6)==0)||(Weight_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,6)==0)
                                right_block_weights=[2,4];
                            else
                                right_block_weights=[2,4,6];
                            end
                            
                            %perform weight balance crossover
                            balance_factor=rand;
                            weight_stash=Weight_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter,right_block_weights);
                            Weight_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter,right_block_weights)=(balance_factor*weight_stash)+((1-balance_factor)*Weight_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,right_block_weights));
                            Weight_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,right_block_weights)=((1-balance_factor)*weight_stash)+(balance_factor*Weight_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,right_block_weights));
                            
                            %biases
                            balance_factor=rand;
                            bias_stash=Bias_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter,2);
                            Bias_Population(Winner_Index_Vector(crossover_operation_counter),row_counter,column_counter,2)=(balance_factor*bias_stash)+((1-balance_factor)*Bias_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,2));
                            Bias_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,2)=((1-balance_factor)*bias_stash)+(balance_factor*Bias_Population(Winner_Index_Vector(crossover_operation_counter+1),row_counter,column_counter,2));
                        end
                    end
                end
            end
        end
    end
          
    %%%mutation
    %Note: This section tunes SignalFlow_Population, Weight_Population,
    %and Bias_Population
    for mutation_operation_counter=1:DeletionNumber
        
        %perform one point mutation in 2D space for structure of block
        %based neuran network
        if rand<=Structure_MutationRate
            %select random position in block based neural network
            random_row=randi(NetworkRows);
            random_column=randi(NetworkColumns-1);
            
            %toggle selected signal flow
            SignalFlow_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column+1,1)=SignalFlow_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column,2);
            SignalFlow_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column,2)=abs(1-SignalFlow_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column,2));
        end
        
        %perform one point mutation in 2D space for weights or biases of 
        %blocks
        if rand<=WeightBias_MutationRate
            %select random block
            random_row=randi(NetworkRows);
            random_column=randi(NetworkColumns);
            
            %select an appropriate random weight or bias
            random_WeightBias=randi(9);
            
            %detect that the selected one is bias or weight
            if random_WeightBias>6
                random_WeightBias=random_WeightBias-6;
                %check if the selected bias exists (it is non zero)
                while Bias_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column,random_WeightBias)==0
                    random_WeightBias=randi(3);
                end
                
                %perform mathematical mutation on the selected bias
                Bias_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column,random_WeightBias)=Bias_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column,random_WeightBias)+random('normal',0,1);
            else
                %check if the selected weight exists (it is non zero)
                while Weight_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column,random_WeightBias)==0
                    random_WeightBias=randi(6);
                end
                
                %perform mathematical mutation on the selected weight
                Weight_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column,random_WeightBias)=Weight_Population(Winner_Index_Vector(mutation_operation_counter),random_row,random_column,random_WeightBias)+random('normal',0,1);
            end
            
        end
    end
    
    %%%fitness estimation
    for offspring_counter=1:DeletionNumber
        
        [Fitness_Matrix(1,Winner_Index_Vector(offspring_counter)), Weight_Population(Winner_Index_Vector(offspring_counter),:,:,:), Bias_Population(Winner_Index_Vector(offspring_counter),:,:,:), SignalFlow_Population(Winner_Index_Vector(offspring_counter),:,:,:)] = FitnessFunction(Example_Matrix, Example_Number, NetworkRows, NetworkColumns, SignalFlow_Population(Winner_Index_Vector(offspring_counter),:,:,:), Weight_Population(Winner_Index_Vector(offspring_counter),:,:,:), Bias_Population(Winner_Index_Vector(offspring_counter),:,:,:), FunctionType, Slope_Parameter, FunctionAmplitude, ComparisonType, OutputNumber);
        
        %check for best found
        if Fitness_Matrix(1,Winner_Index_Vector(offspring_counter))>=Best_Fitness
            Best_Fitness=Fitness_Matrix(1,Winner_Index_Vector(offspring_counter));
            Best_SignalFlow_Matrix=SignalFlow_Population(Winner_Index_Vector(offspring_counter),:,:,:);
            Best_Weight_Matrix=Weight_Population(Winner_Index_Vector(offspring_counter),:,:,:);
            Best_Bias_Matrix=Bias_Population(Winner_Index_Vector(offspring_counter),:,:,:);            
        end
    end
                    
    %%%replacement
    %detecting worst chromosomes
    [sorted_fitness_matrix, Sorted_Fitness_Indices]=sort(Main_Fitness_Matrix);
    
    %performing replacement
    Main_SignalFlow_Population(Sorted_Fitness_Indices(1:DeletionNumber),:,:,:)=SignalFlow_Population(Winner_Index_Vector,:,:,:);
    Main_Weight_Population(Sorted_Fitness_Indices(1:DeletionNumber),:,:,:)=Weight_Population(Winner_Index_Vector,:,:,:);
    Main_Bias_Population(Sorted_Fitness_Indices(1:DeletionNumber),:,:,:)=Bias_Population(Winner_Index_Vector,:,:,:);
    Main_Fitness_Matrix(1,Sorted_Fitness_Indices(1:DeletionNumber))=Fitness_Matrix(1,Winner_Index_Vector);
    
end

end

