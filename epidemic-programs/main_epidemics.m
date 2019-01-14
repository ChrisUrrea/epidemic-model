% set random seed for replication
rng(007);
% Save feature values 10 time steps for 9,000 train epidemics 
% -> 90,000 timesteps and 14 features per time step
train_features = zeros(90000, 14);
% Save epidemic parameters / target variables
% That is the beta, alpha, delta used to simulate each epidemic
train_targets = zeros(9000, 3);

% same logic for test data structures
test_features = zeros(10000, 14);
test_targets = zeros(1000, 3);

% simulate the full epidemics!!
for j = 1:10000
    % randomly instantiate the epidemic's infection rate
    % beta in the range of [0.15 - 0.8]
    beta = (.65)*rand(1,1) +.15; % setting floor at .15
     % randomly instantiate the epidemic's recovery rate
    % in the range of [.1 - (alpha-0.05)]
    bmax = beta - .15;
    alpha = (bmax)*rand(1,1) + .1; % setting floor at .1
    % randomly instantiate the epidemic's death rate
    % delta in the range of [.01 - .04]
    delta = (.04)*rand(1,1) + .01;
    % randomize our starting infected as an integer in range of [44 - 55]
    I = 45:55;    
    init_infected = I(randi([1,numel(I)]));
    % fix initial susceptible population at 100k
    init_pop = 100000;
    seed = j;
    % simulate the epidemic and return row of features as observation
    observation = SIR_epidemic_simulation(beta, alpha, delta, init_pop, init_infected, seed);
    % index handlers for start:end 10 time step window

    if j <= 9000 % save as training simulation
        start_index = ((j-1) * 10) + 1;
        end_index = j*10;
        % save features to data by the start:end window size
        train_features(start_index:end_index, :) = observation;
        % save corresponding beta, alpha, delta params of the epidemic simulation
        train_targets(j,1) = beta;
        train_targets(j,2) = alpha;
        train_targets(j,3) = delta;
    else % save as a test simulation
        k = j-9000;
        start_index = ((k-1) * 10) + 1;
        end_index = k*10;
        % save features to data by the start:end window size
        test_features(start_index:end_index, :) = observation;
        % save corresponding beta, alpha, delta params of the epidemic simulation
        test_targets(k,1) = beta;
        train_targets(k,2) = alpha;
        train_targets(k,3) = delta;
    end
end
% save our full training data as csv files to model in python
csvwrite('9k-features-train.csv', train_features);
csvwrite('9k-targets-train.csv', train_targets);
% save our full testing data as csv files to model in python
csvwrite('1k-features-test.csv', test_features);
csvwrite('1k-targets-test.csv', test_targets);