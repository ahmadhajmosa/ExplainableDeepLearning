import numpy as np
import pandas as pd


def random_data(n_vars, size):
    # n_vars.........number of variables (columns)
    # size...........number of observations (rows)

    # generate variable strings
    variables = list()
    variable_df = list()
    for i in range(n_vars):
        variables.append('X' + str(i) + '= ')
        variable_df.append('X' + str(i))

    # empty array
    data = np.empty((0, 0), dtype=object)
    data_df = pd.DataFrame(columns=variable_df)

    # for each row, generate the string with random values of 0 and 1
    for i in range(size):

        # initialize empty string and empty data frame row
        string = ''
        row = []

        # iterator to know when string end and no further commas are needed
        var_iteration = 0

        # for each variable, generate a value of 0 or 1 and append to the string
        for var in variables:
            number = np.random.choice([0, 1], size=1)[0]

            string = string + var + str(number) + ' '
            row.append(number)

            var_iteration += 1

        # convert the row (which is still a list) to a dataframe row
        row_df = pd.DataFrame([row], columns=variable_df)

        # append the string to the dataset and the row to the dataframe
        data = np.append(data, string)
        data_df = data_df.append(row_df, ignore_index=True)

    return data, data_df


def random_rules(n, size, operator='logical', brackets=True):
    # n..............number of rules to generate
    # size...........number of columns to draw from
    # operator_list..type of operator to use (takes values from [logical, numerical])
    # brackets.......should there be brackets in the rules?

    logical = ["&", "|"]
    numerical = ["+", "-", "*", "/"]

    if operator == 'logical':
        operator_list = logical
    elif operator == 'numerical':
        operator_list = numerical
    else:
        raise ValueError('Wrong operator value passed')

    # generate variable list
    var_list = list()
    for i in range(size):
        #var_list.append("data['X" + str(i) + "']")
        #var_list.append("~data['X" + str(i) + "']")
        var_list.append("X" + str(i))
        var_list.append("~X" + str(i))
    # generate size of each rule (how many columns are used)
    size_list = np.random.choice(np.arange(2, 4), size=n, replace=True).tolist()

    # generate rules
    # empty list for the rules
    rules = [''] * n
    # iterator for counting and subsetting
    rule_iteration = 0
    input_variable_list = []
    # loop over the generated sizes
    for rule_size in size_list:



        # copy the full variable list so we start each rule with the full set of variables
        var_list_update = var_list.copy()
        # iterator for counting and to know when to end the rule (so no operation is added at the end)
        var_iteration = 0

        for i in range(rule_size):

            # if the rule is too long, break and return the finished rule
            if var_iteration >= rule_size:
                break

            # generate if there is a bracket
            if brackets:
                bracket = np.random.choice([0, 1], size=1, p=[0.8, 0.2])[0]
            else:
                bracket = 0

            # if there should be a bracket, run this
            if bracket == 1:

                # choose random variables (2 for the whole bracket expression)
                var_index = np.random.choice(len(var_list_update), replace=False, size=2)
                variable1 = var_list_update[var_index[0]]
                variable2 = var_list_update[var_index[1]]
                input_variable_list.append(variable1)
                input_variable_list.append(variable2)


                # choose random operation
                op = np.random.choice(operator_list, size=1)[0]

                # append the whole bracket to the rule
                rules[rule_iteration] = rules[rule_iteration] + '( ' + variable1 + ' ' + op + ' ' + variable2 + ' ) '

                # if the rule is still not long enough, add another operation and continue generating
                if var_iteration < rule_size - 2:
                    op2 = np.random.choice(operator_list, size=1)[0]
                    rules[rule_iteration] = rules[rule_iteration] + op2 + ' '

                # delete used variables from variable list (only for this rule)
                var_index = np.sort(var_index)
                del var_list_update[var_index[1]]
                del var_list_update[var_index[0]]

                var_iteration += 2

            else:

                # choose random variable
                var_index = np.random.choice(len(var_list_update), size=1)[0]
                variable = var_list_update[var_index]
                input_variable_list.append(variable)
                # choose random operation
                op = np.random.choice(operator_list, size=1)[0]

                # as long as the rule is not ending, we insert the variable and the operation, when the last entry
                # for the rule is drawn, only the variable is added, without an operation
                if var_iteration < rule_size - 1:
                    rules[rule_iteration] = rules[rule_iteration] + variable + ' ' + op + ' '
                else:
                    rules[rule_iteration] = rules[rule_iteration] + variable

                # delete used variable from variable list (only for this rule)
                del var_list_update[var_index]

                var_iteration += 1

        rule_iteration += 1
        #print(input_variable_list)
        #print('not',[sel for sel in np.random.permutation((var_list_update))[:3] if '~' not in sel])
        selected_target=[sel for sel in np.random.permutation((var_list_update)) if '~' not in sel]
    return rules, selected_target
