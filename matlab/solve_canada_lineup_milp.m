function solve_canada_lineup_milp()
% SOLVE_CANADA_LINEUP_MILP
%
% Solves the knapsack lineup optimization problem:
%
%   max   sum_j beta_j x_j
%   s.t.  sum_j w_j x_j <= 8
%         sum_j x_j     = 4
%         x_j in {0,1}
%
% where:
%   beta_j = estimated player impact (GD/min)
%   w_j    = player classification rating
%   x_j    = binary decision variable (player selected)
%
% This implementation is intentionally kept lecture-exact,
% but augmented with diagnostics for reporting and analysis.

    fprintf("\n--- Solving Canada Lineup MILP (Knapsack) ---\n");

    %% ----------------------------------------------------
    % Load data exported from Python
    %% ----------------------------------------------------
    impacts_tbl = readtable("player_impacts.csv");   % player, impact
    ratings_tbl = readtable("player_ratings.csv");   % player, rating

    player_names = impacts_tbl.player;
    impacts = impacts_tbl.impact;     % beta_j
    ratings = ratings_tbl.rating;     % w_j

    n = length(impacts);

    fprintf("Number of candidate players: %d\n", n);

    %% ----------------------------------------------------
    % Decision variables
    % x_j = 1 if player j is selected
    %% ----------------------------------------------------
    % intlinprog minimizes, so we negate the objective
    f = -impacts;

    intcon = 1:n;          % all variables are integer
    lb = zeros(n,1);       % x_j >= 0
    ub = ones(n,1);        % x_j <= 1

    %% ----------------------------------------------------
    % Constraints
    %% ----------------------------------------------------

    % (1) Exactly 4 players selected
    Aeq = ones(1,n);
    beq = 4;

    % (2) Classification constraint
    % sum_j w_j x_j <= 8
    A = ratings';
    b = 8;

    %% ----------------------------------------------------
    % Solve MILP
    %% ----------------------------------------------------
    options = optimoptions( ...
        'intlinprog', ...
        'Display','off' ...
    );

    [x, opt_value, exitflag, output] = intlinprog( ...
        f, intcon, A, b, Aeq, beq, lb, ub, options);

    if exitflag <= 0
        error("MILP failed to find a feasible solution.");
    end

    fprintf("MILP solved successfully.\n");
    fprintf("Solver status: %s\n", output.message);

    %% ----------------------------------------------------
    % Post-solution analysis (report-friendly)
    %% ----------------------------------------------------

    selected_idx = find(x > 0.5);
    selected_players = player_names(selected_idx);

    total_rating = sum(ratings(selected_idx));
    total_impact = sum(impacts(selected_idx));

    classification_slack = b - total_rating;

    %% ----------------------------------------------------
    % Display interpretable results
    %% ----------------------------------------------------
    fprintf("\n--- Optimal Lineup ---\n");
    disp(selected_players);

    fprintf("Total impact (objective value): %.4f GD/min\n", total_impact);
    fprintf("Total classification points used: %.2f / 8\n", total_rating);
    fprintf("Unused classification capacity (slack): %.2f\n", classification_slack);

    %% ----------------------------------------------------
    % Save results for app / report
    %% ----------------------------------------------------
    solution_tbl = table( ...
        player_names, x, ...
        'VariableNames', {'player','selected'} ...
    );

    writetable(solution_tbl, "optimal_lineup.csv");

    summary_tbl = table( ...
        total_impact, total_rating, classification_slack, ...
        'VariableNames', {'total_impact','total_rating','rating_slack'} ...
    );

    writetable(summary_tbl, "milp_summary.csv");

    fprintf("\nResults written to:\n");
    fprintf("  - optimal_lineup.csv\n");
    fprintf("  - milp_summary.csv\n");
    fprintf("-----------------------------------------------\n\n");

end
